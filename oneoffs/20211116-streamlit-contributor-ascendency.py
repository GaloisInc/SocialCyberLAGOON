
from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import altair as alt
import arrow
import collections
import functools
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
import re
import scipy.signal
import sqlalchemy as sa
import streamlit as st
import time
import torch
import wordcloud

st.set_page_config(layout="wide")

## See bottom of script for page selection

cache_dec = lambda: st.cache(allow_output_mutation=True,
        hash_funcs={'sqlalchemy.orm.session.Session': lambda x: 'hi',
            'sqlalchemy.orm.attributes.InstrumentedAttribute': lambda x: 'hi'})

@cache_dec()
def entity_obs_hops(sess, entity_id, *args, **kwargs):
    ent = sess.query(sch.FusedEntity).where(sch.FusedEntity.id == entity_id).scalar()
    result = ent.obs_hops(*args, **kwargs)

    # Normally, when a session exits, all information is wiped. If we expunge
    # the object first, it isn't. So, expunge all objects we might want to query
    # downstream.
    for r in result:
        for ro in [r, r.src, r.dst]:
            if ro in sess:
                sess.expunge(ro)

    return result


def density_convolve(x, pts, std):
    '''
    Args:
        x: Data points at which density is observed.
        pts: Time line to project density onto.
        std: Standard deviation for convolution.
    '''
    d = pts[1] - pts[0]
    x = torch.tensor(x, dtype=pts.dtype)
    x_arr = torch.zeros_like(pts)
    x_arr.scatter_add_(0,
            ((x - pts[0]) / d).to(torch.long).clamp_(max=pts.size(0)-1),
            torch.ones_like(x))
    npts = int(2 * 4 * std / d)
    if npts % 2 == 0:
        npts += 1
    y = scipy.signal.convolve(
            x_arr,
            scipy.signal.gaussian(npts, std / d))
    return y[npts // 2:-(npts // 2)]
    #x['peps'].sum() * scipy.stats.gaussian_kde(
    #x['date'],
    #bw_method=pts,
    #).evaluate(extent_pts)


def violin(df, sort_order=None):
    '''expects 'date', 'density', and 'collab' columns.

    Optionally, 'density_color' for a bi-color effect.
    '''
    c = (alt.Chart(df)
            #.transform_density(
            #    'date',
            #    bandwidth=1*365*24*3600*1e3,
            #    as_=['date', 'density'],
            #    extent=extent,
            #    groupby=['collab'])
            # We want to keep scale, so adjust a bit
            .mark_area(orient='vertical').encode(
                x='date:T',
                color=alt.Color(
                    'collab:N' if 'density_color' not in df else 'density_color:N',
                    title=None if 'density_color' not in df else 'PEP status',
                    legend=None if 'density_color' not in df else alt.Legend(),
                ),
                y=alt.Y('density:Q', stack='center',
                    impute=None, title=None, axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True)),
                row=alt.Row('collab:N',
                    sort=sort_order,
                    header=alt.Header(
                        labelFontSize=12,
                        labelColor='white', labelAngle=0, labelOrient='right',
                        labelAnchor='end', labelBaseline='bottom',
                        titleColor='white', titleOrient='left',
                    )),
            )
            .configure_view(height=50, stroke=None)
            .configure_facet(spacing=0)
            )
    st.altair_chart(c, use_container_width=True)


def inlay_code(sess, entity_id):
    """Make a violin density plot of all unique files that this contributor
    touched.
    """
    ent = sess.query(sch.FusedEntity).where(sch.FusedEntity.id == entity_id).scalar()
    st.text('Plot = number unique files touched over time')

    f1 = sa.orm.aliased(sch.FusedObservation)
    e2 = sa.orm.aliased(sch.FusedEntity)
    f2 = sa.orm.aliased(sch.FusedObservation)
    e3 = sa.orm.aliased(sch.FusedEntity)

    # Step 1 -- find all user commits
    commits_direct = (sess.query(e2.id)
            .select_from(f1)
            .where(f1.src_id == entity_id)
            .where(f1.type.in_([sch.ObservationTypeEnum.created,
                sch.ObservationTypeEnum.committed]))
            .join(e2, e2.id == f1.dst_id)
            .where(e2.type == sch.EntityTypeEnum.git_commit))
    # Step 2 -- find all direct PRs
    pr_direct = (sess.query(e2.id)
            .select_from(f1)
            .where(f1.src_id == entity_id)
            .where(f1.type == sch.ObservationTypeEnum.created)
            .join(e2, e2.id == f1.dst_id)
            .where(e2.type == sch.EntityTypeEnum.github_pullrequest)
    )
    # Step 3 -- go from PRs to commits (only works if commits were merged in...)
    pr_subq = pr_direct.subquery()
    commits_pr = (sess.query(e2.id)
            .select_from(pr_subq)
            .join(f1, f1.dst_id == pr_subq.c.id)
            .where(f1.type == sch.ObservationTypeEnum.attached_to)
            .join(e2, f1.src_id == e2.id)
            .where(e2.type == sch.EntityTypeEnum.git_commit))

    # Finally, go from commits to time+file touched
    commits_all = sa.select(commits_pr.subquery()).union(commits_direct).subquery()
    file_map = (sess.query(f1.time, e2.name)
            .select_from(commits_all)
            .join(f1, f1.src_id == commits_all.c.id)
            .where(f1.type == sch.ObservationTypeEnum.modified)
            .join(e2, f1.dst_id == e2.id)
            .where(e2.type == sch.EntityTypeEnum.file)
            ).all()

    df = pd.DataFrame(file_map, columns=['date', 'file'])
    df['date'] = df['date'].apply(lambda x: x.timestamp()*1e3)
    extent = [arrow.get('1990-01-01').timestamp()*1e3,
            arrow.get('2020-01-01').timestamp()*1e3]
    extent_pts = torch.linspace(extent[0], extent[1], 101)
    density_std = .5*365*24*3600*1e3
    df = (
            df.groupby('file')
            .apply(lambda x: pd.DataFrame({
                'date': extent_pts,
                'density': density_convolve(x['date'].values, extent_pts,
                    std=density_std),
            }))
            .reset_index()
    )
    one_kernel = density_convolve(torch.tensor([0.]),
            torch.tensor([-density_std, 0, density_std]),
            std=density_std).max()
    df['density'] = df['density'].clip(upper=one_kernel)
    df = df.groupby('date').sum().reset_index()
    df['collab'] = ent.name
    violin(df)


def page_contributor_detail(sess, entity_id):
    # Grab an entity
    ent = sess.query(sch.FusedEntity).where(sch.FusedEntity.id == entity_id).scalar()
    if ent is None:
        return

    st.header(ent.name)
    bar = st.progress(0)
    year_start = 1995
    year_end = 2020

    # Partition by year
    rows = []
    for year in range(year_start, year_end):
        bar.progress((year - year_start) / (year_end - year_start))

        row = {'year': year}
        rows.append(row)
        subnet = entity_obs_hops(sess, entity_id, 2,
                time_min=arrow.get(year=year, month=1, day=1).datetime,
                time_max=arrow.get(year=year+1, month=1, day=1).datetime)
        row['subnet_size'] = len(subnet)

        src = collections.defaultdict(list)
        counters = collections.defaultdict(int)
        tox_seen = set()
        for s in subnet:
            src[s.src.id].append(s)

            for o in [s.src, s.dst]:
                if o.id in tox_seen:
                    continue
                tox_seen.add(o.id)

                tox = 0.
                for k, v in o.attrs.items():
                    if k.startswith('computed_badwords_'):
                        tox += v
                if s.src.id == entity_id or s.dst.id == entity_id:
                    counters['tox_self'] += tox
                else:
                    counters['tox_2hop'] += tox

        for obs in src[ent.id]:
            counters[f'{obs.type.name} -> {obs.dst.type.name}'] += 1

        for obs in src[ent.id]:
            if (obs.type.name != sch.ObservationTypeEnum.created.name
                    or obs.dst.type.name != sch.EntityTypeEnum.pep.name):
                continue
            a = obs.dst.attrs
            counters[f"peps {repr(a['status'])}"] += 1
            if a['status'] in ['Final', 'Superseded', '']:
                counters[f'peps successful'] += 1

        if 'created -> pep' in counters:
            # Log space re-frame
            counters['peps successful'] = (
                    1 + 9 *
                    counters['peps successful'] / counters['created -> pep'])

        for k, v in counters.items():
            row[k] = v

    df = pd.DataFrame(rows)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.set_index(df.year)
    st.table(df)

    # Draw a plot
    st.pyplot(p9.ggplot.draw(
        p9.ggplot(df)
        + p9.aes(x='year')
        + p9.geom_line(p9.aes(y='subnet_size', color='"subnet_size"'))
        + p9.geom_line(p9.aes(y='tox_self', color='"tox_self"'))
        + p9.geom_line(p9.aes(y='tox_2hop', color='"tox_2hop"'))
        + p9.geom_line(p9.aes(y='created -> pep', color='"created peps"'))
        + p9.geom_line(p9.aes(y='peps successful', color='"peps successful"'))
        + p9.scale_y_log10()
    ))

    ## Word clouds

    columns = st.columns(3)
    col_idx = 0

    for year in range(year_start, year_end):
        subnet = entity_obs_hops(sess, entity_id, 2,
                time_min=arrow.get(year=year, month=1, day=1).datetime,
                time_max=arrow.get(year=year+1, month=1, day=1).datetime)
        words = []
        n_msg = 0
        for obs in subnet:
            if obs.src.id != entity_id:
                continue
            if obs.dst.type.name != sch.EntityTypeEnum.message.name:
                continue

            n_msg += 1
            try:
                text = obs.dst.attrs['body_text']
            except KeyError:
                raise KeyError(f'no body_text in ID {obs.dst.id}')
            text_lines = []
            last_line = None
            for line in text.split('\n'):
                line = line.strip()
                if re.search(r'^(On .*,.*wrote|Le mer\..*crit ):$', line) is not None:
                    continue
                if len(text_lines) == 0 and re.search(r'(schrieb|wrote):$', line, flags=re.I) is not None:
                    # Another header
                    continue
                if line.startswith('>'):
                    continue
                text_lines.append(line)

                # Strip off things that look like a footer
                if re.search(r'^(cheers|regards|thank you|thanks|--),?$', line,
                        flags=re.I) is not None:
                    last_line = len(text_lines) - 1
                    # FIXME
                    break
            if last_line is not None:
                text_lines = text_lines[:last_line]
            words.extend(text_lines)
            # End of message marker
            words.append(',,,')
        words = ' '.join(words)

        if words.strip():
            col = columns[col_idx]
            col_idx = (col_idx + 1) % len(columns)
            with col:
                st.write(f'## {year} - {n_msg} messages')
                wc = wordcloud.WordCloud(
                        stopwords=wordcloud.STOPWORDS.union([
                            'guido', 'van', 'rossum',
                            'code',
                            'home', 'home page',
                            'one', 'org', 'page',
                            'python', 'python org', 'python dev',
                            'think', 'use', 'will',
                        ])
                        ).generate(words)
                fig = plt.figure()
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
                plt.close()

    bar.progress(1.)


def page_pep_groups(sess, entity_id):
    only_authored = st.checkbox('Only PEPS from this author')

    if only_authored:
        pep_author = sess.get(sch.FusedEntity, entity_id)
        peps = (sess.query(sch.FusedEntity)
                .where(sch.FusedEntity.type == sch.EntityTypeEnum.pep)
                .join(sch.FusedObservation, sch.FusedEntity.obs_as_dst)
                .where(sa.orm.with_parent(pep_author, sch.FusedEntity.obs_as_src))
                .order_by(sch.FusedEntity.name)
                ).all()
    else:
        peps = (sess.query(sch.FusedEntity)
                .where(sch.FusedEntity.type == sch.EntityTypeEnum.pep)
                .order_by(sch.FusedEntity.name)
                ).all()
    peps_loaded = 0
    author_idx = {}
    author_idx_rev = {}
    pep_collabs = []  # [pep, {people}]
    pep_dates = []
    pep_goods = []
    def get_author(oid):
        if oid not in author_idx:
            author_idx[oid] = len(author_idx)
            author_idx_rev[author_idx[oid]] = sess.query(sch.FusedEntity).where(sch.FusedEntity.id == oid).scalar().name
        return author_idx[oid]

    for p in peps:
        peps_loaded += 1
        collabs = _pep_group_collabs(sess, p.id)
        collabs = [get_author(i) for i in collabs]
        pep_collabs.append(collabs)
        pep_dates.append(arrow.get(p.attrs['created'], 'DD-MMM-YYYY'))
        pep_goods.append(True if p.attrs['status'].lower() in [
                'active', 'accepted', 'final', 'superseded'] else False)
    st.write(f'Loaded {peps_loaded} PEPs')

    # Now have a vector of (pep, people). Make it a matrix, do some math.
    indices = []
    for i, pep_col in enumerate(pep_collabs):
        for j in pep_col:
            indices.append((i, j))
    indices = torch.tensor(indices, dtype=torch.long)
    mat = torch.sparse_coo_tensor(indices.T,
            torch.ones_like(indices[:, 0], dtype=torch.float))
    mat = mat.to_dense()  # Sparse support is terrible
    shared_projects = mat.T @ mat

    for auth_i in [author_idx[entity_id]]:
        if shared_projects[auth_i, auth_i] < 5:
            # Less than 5 peps, ignore this person
            continue

        st.header(author_idx_rev[auth_i])
        inlay_code(sess, entity_id)
        st.text('First plot = PEP involvement over time; second plot set = P(other author involved|this author involved)')

        data = []
        collabs_seen = set()
        for date, good, pep in zip(pep_dates, pep_goods, mat):
            if not pep[auth_i].item():
                continue
            for j in pep.nonzero()[:, 0].tolist():
                if shared_projects[auth_i, j] < max(1, shared_projects[auth_i, auth_i] * 0.1):
                    # Ignore potential collaborators which only collaborated on
                    # one pep, or less than a third of all.
                    continue
                collabs_seen.add(author_idx_rev[j])
                data.append({'collab': author_idx_rev[j], 'peps': 1,
                    'peps_good': 1 if good else 0,
                    'date': date.timestamp()*1e3})

        df = pd.DataFrame(data)
        # Manual, we want a density estimation scaled by count
        extent = [arrow.get('1990-01-01').timestamp()*1e3,
                arrow.get('2020-01-01').timestamp()*1e3]
        extent_pts = torch.linspace(extent[0], extent[1], 101)

        # compute P(Other & Author)
        density_std = .5*365*24*3600*1e3
        df = (df
                .groupby('collab')
                .apply(lambda x: pd.DataFrame({
                    'date': extent_pts,
                    'density': density_convolve(x['date'].values, extent_pts,
                        std=density_std),
                    'density_good': density_convolve(
                        x['date'].values[x['peps_good'].values != 0],
                        extent_pts, std=density_std),
                }))
                .reset_index())

        # Make one chart with just P(Author), then P(Other|Author)
        p_auth = df[df['collab'] == author_idx_rev[auth_i]]
        # Old version, does not discriminate good vs bad
        #violin(p_auth)

        # Add a chart on P(success)
        df2 = p_auth.copy()
        df2['density_color'] = 'good'
        df2['density'] = df2['density_good']
        df3 = p_auth.copy()
        df3['density_color'] = 'bad'
        df3['density'] = df3['density'] - df3['density_good']
        violin(pd.concat([df2, df3]))

        # Second chart -- divide out p_auth involvement
        df = df[df['collab'] != author_idx_rev[auth_i]]
        # First, sort by involvement alongside p_auth
        sort_order = df.groupby('collab')['density'].sum().sort_values(ascending=False).index.values

        # Then divide out
        df['density'] = df['density'] / df['date'].map(p_auth.set_index('date')['density'])
        violin(df, sort_order)
    return

    # Old bar chart code
    authors = mat.T @ mat

    for auth_i in range(len(author_idx)):
        auth_row = authors[auth_i]
        st.header(author_idx_rev[auth_i])
        vals, indx = auth_row.sort(descending=True)
        data = []
        for i, v in zip(indx, vals):
            i = i.item()
            v = v.item()
            if v <= max(1, authors[auth_i, auth_i].item() * 0.3):
                break
            data.append({'name': author_idx_rev[i], 'collab': v})
        df = pd.DataFrame(data)

        c = alt.Chart(df).mark_bar().encode(
                y=alt.Y('name', sort='-x'),
                x='collab:Q')
        st.altair_chart(c, use_container_width=True)

        if auth_i > 5:
            break


@cache_dec()
def _pep_group_collabs(sess, p_id):
    """For a given PEP with entity ID `p_id`, returns a set of all collaborators
    who created or messaged in reference to this PEP.
    """
    collabs = set()
    neighborhood = entity_obs_hops(sess, p_id, 1)
    for obs in neighborhood:
        if (obs.type.name == sch.ObservationTypeEnum.created.name
                and obs.dst.id == p_id):
            collabs.add(obs.src.id)
        elif (obs.type.name == sch.ObservationTypeEnum.message_ref.name
                and obs.dst.id == p_id):
            talker = (sess.query(sch.FusedObservation)
                    .where(sa.orm.with_parent(obs.src, sch.FusedEntity.obs_as_dst))
                    .where(sch.FusedObservation.type == sch.ObservationTypeEnum.message_from)
                    ).all()
            # Not all have FROM...
            #assert len(talker) >= 1, f'{obs.src.id} -- {repr(talker)}'
            for talk in talker:
                collabs.add(talk.src.id)
    return collabs


page_opts = {
        'Contributor detail': dict(func=page_contributor_detail),
        'PEP Organizational Groups': dict(func=page_pep_groups),
}
page_value = st.sidebar.selectbox('Navigation', page_opts.keys())
page = page_opts[page_value]
st.header(page_value)
with get_session() as sess:
    st.caption('Well-known user ids:')
    st.text('Nick Coghlan 131166\nGuido van Rossom 130975\nVictor Stinner 130913\nTerry Jan Reedy 2488888\nThomas Heller 2490207\nNed Deily 2488863\nRaymond Hettinger 2488920')
    st.text('Moshe Zadka 2940280\nSerhiy Storchaka 2488862\nJim Jewett 2510279')
    entity_id = st.number_input('Entity ID', value=131166)

    page['func'](sess, entity_id)
    # Avoid saving anything
    sess.rollback()

