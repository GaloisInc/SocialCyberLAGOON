"""Entity fusion program.
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.db.temptable import (temp_table_from_model, multi_session_dropper,
        multi_session_remap)

import multiprocessing
import random
import re
import sqlalchemy as sa
import time
import tqdm

Base = sa.orm.declarative_base()
class EntitySearch(Base):
    __tablename__ = 'EntitySearch_fusion'

    id: int = sa.Column(sa.Integer, primary_key=True)
    name: str = sa.Column(sa.String(65534), index=True)
    email: str = sa.Column(sa.String(65534), index=True)
    github_user: str = sa.Column(sa.String(65534), index=True)

    name_idx: str = sa.Column(sa.String, sa.Computed('fusion_name_munge(name)'))

    __table_args__ = (
        sa.Index('idx_entitysearch_zz', 'name_idx',
            postgresql_ops={'name_idx': 'gist_trgm_ops'},
            postgresql_using='gist'),
    )


def fusion_compute():
    """Re-computes all fused entities, respecting manual annotations.
    """
    # IMPORTANT -- allocate multiprocessing before using EntitySearch in a
    # query.
    with multiprocessing.Pool() as p:
        _fusion_compute(p)


def _fusion_compute(p):
    with get_session() as sess:
        # metaphone() by default ignores spaces, so use array magic to count
        # them.
        sess.execute(sa.text(r'''
                create or replace function fusion_name_munge(text)
                    returns text as $$
                select
                    array_to_string(array_agg(metaphone(part, 20)), ' ')
                    FROM unnest(string_to_array($1, ' ')) AS part
                $$ language sql immutable'''))

        ETable = temp_table_from_model(sess, EntitySearch, multi_session=True)

    with multi_session_dropper(get_session, ETable, keep=False):
        with get_session() as sess:
            print(f'Gather data on community members')
            sess.execute(sa.insert(ETable).from_select(['id', 'name', 'email', 'github_user'],
                    sa.select(sch.Entity.id,
                        # NOTE -- may be NULL!
                        sa.func.lower(sch.Entity.attrs['name'].astext),
                        sa.func.lower(sch.Entity.attrs['email'].astext),
                        sa.func.lower(sch.Entity.attrs['github_user'].astext),
                    )
                    .where(
                        (sch.Entity.type == 'person') & (sa.func.length(sch.Entity.name) < 1000))
                    ))

            all_obj_ids = [o[0] for o in sess.execute(sa.select(ETable.id))]

            # Purge previous fusion results
            print(f'Purging previous fusion')
            sess.execute(sa.delete(sch.EntityFusion))

            # Fuse commits
            print(f'Fusing commits')
            cte = (sess.query(sch.Entity.attrs['commit_sha'].astext.label('sha'), sa.func.min(sch.Entity.id).label('id'))
                    .where(sch.Entity.type == 'git_commit')
                    .group_by(sch.Entity.attrs['commit_sha'].astext)).cte()
            sess.execute(sa.insert(sch.EntityFusion).from_select(['id_lowest', 'id_other'],
                    sa.select(cte.c.id, sch.Entity.id)
                    .select_from(cte)
                    .join(sch.Entity, sch.Entity.attrs['commit_sha'].astext == cte.c.sha)))

        # Merge every entity
        list(tqdm.tqdm(p.imap(_fuse_entity, all_obj_ids), total=len(all_obj_ids),
                desc='Fusing people'))

        with get_session() as sess:
            # Anything that's not a person and not yet merged is exempt from
            # fusion
            print(f'Fusing everything else')
            sess.execute(sa.insert(sch.EntityFusion).from_select(['id_lowest', 'id_other'],
                    sa.select(sch.Entity.id, sch.Entity.id.label('id2'))
                    .where(
                        ~sa.select(sch.EntityFusion).where(sch.EntityFusion.id_other == sch.Entity.id).exists()
                    )))


def _fuse_entity(entity_id):
    # Multiprocessing hack
    ET = multi_session_remap(EntitySearch)

    # Conflicts expected. Basically, our primary_key constraint will be
    # invalidated, and we should try again
    while True:
        try:
            with get_session() as sess:
                obj = sess.execute(sa.select(ET).where(ET.id == entity_id)).scalar()

                added_to_group = False
                def add_to_group(id1, id2, comment):
                    """Merge the groups specified by id1 and id2."""
                    nonlocal added_to_group
                    added_to_group = True
                    # We avoid a recursive CTE here because of the unique constraint
                    # on "other"
                    group_membership = sess.execute(sa.select(sch.EntityFusion)
                            .where(sch.EntityFusion.id_other == id1)).scalar()
                    if (group_membership is not None
                            and group_membership.id_lowest != id1):
                        return add_to_group(group_membership.id_lowest, id2,
                                comment)
                    group_membership = sess.execute(sa.select(sch.EntityFusion)
                            .where(sch.EntityFusion.id_other == id2)).scalar()
                    if (group_membership is not None
                            and group_membership.id_lowest != id2):
                        return add_to_group(id1, group_membership.id_lowest,
                                comment)

                    # Both id1 and id2 are the lowest ids in their respective
                    # groups.
                    low = min(id1, id2)
                    high = max(id1, id2)

                    # Whichever is the higher, if it's the lowest of one set,
                    # we need to redirect that so that we're the lowest
                    supplant_group = False
                    if low != high:
                        supplant_group = sess.query(sa.select(sch.EntityFusion)
                                .where(sch.EntityFusion.id_lowest == high).exists()
                                ).scalar()
                        if supplant_group:
                            sess.execute(sa.update(sch.EntityFusion)
                                    .where(sch.EntityFusion.id_lowest == high)
                                    .values(id_lowest=low))
                            # Note -- there is now link of lowest:lowest, so
                            # existing_group should be None if we supplanted

                    # See if this link already exists
                    existing_group = sess.query(sa.select(sch.EntityFusion)
                            .where(
                                (sch.EntityFusion.id_lowest == low)
                                & (sch.EntityFusion.id_other == high)).exists()
                            ).scalar()
                    if existing_group:
                        # Already created
                        return

                    # Lowest did NOT have an entry, so this is safe
                    sess.execute(sa.insert(sch.EntityFusion)
                            .values(id_lowest=low, id_other=high,
                                comment=comment))

                # Same, valid email?
                # FIXME dynamic -- at the moment this is Python-fixed
                banned_list = {
                        'a@b.c',
                        'bogus@does.not.exist.com',
                        'invalid@invalid.invalid',
                        'john@doe.com',
                        'mail.python.org@marco.sulla.e4ward.com',
                        'me@privacy.net',
                        'python-url@phaseit.net',
                        'python@python.org',
                        'python-dev@python.org',
                        'python-help@python.org',
                        'python-list@python.org',
                        'support@superhost.gr',
                        'user@compgroups.net/',

                        # Wildcard list of bad domains
                        '*@none.com',
                        '*@nospam.com',
                        '*@nospam.invalid',
                        '*@null.com',
                        '*@spam.com',
                }
                if (obj.email
                        and re.sub(r'^.*?@', '*@', obj.email) not in banned_list
                        and obj.email not in banned_list
                        and re.search(r'^.*@.*(?<!example)\..*$', obj.email) is not None):
                    same = sess.execute(sa.select(ET).where(
                            (ET.email == obj.email)
                            & (ET.id < obj.id)
                            ).order_by(ET.id).limit(1)).scalar()
                    if same is not None:
                        add_to_group(obj.id, same.id, f'email match: {obj.email}')
                        return

                # Same github_user?
                if obj.github_user:
                    same = sess.execute(sa.select(ET).where(
                            (ET.github_user == obj.github_user)
                            & (ET.id < obj.id)
                            ).order_by(ET.id).limit(1)).scalar()
                    if same is not None:
                        add_to_group(obj.id, same.id, f'github_user match: {obj.github_user}')
                        return

                # Same name? Don't allow for short names
                if obj.name and ' ' in obj.name and len(obj.name) > 5:
                    same = sess.execute(sa.select(ET).where(
                            (ET.name == obj.name)
                            & (ET.id < obj.id)
                            ).order_by(ET.id).limit(1)).scalar()
                    if same is not None:
                        add_to_group(obj.id, same.id, f'name match: {obj.name}')
                        return

                    # Implementation note: postgres only uses the similarity
                    # index correctly when using an ORDER BY clause with no
                    # WHERE clause, where the ORDER BY uses similarity directly
                    # in no arithmetic.
                    sim_dist = ET.name_idx.op('<->')(obj.name_idx)
                    sim_inner = (
                            sa.select(ET.id, ET.name, ET.name_idx,
                                sim_dist.label('sim_dist'))
                            .where(ET.id != obj.id)
                            .where(ET.name_idx.isnot(None))
                            .order_by(sim_dist).limit(1)).subquery()
                    same = sess.execute(sa.select(sim_inner)
                            .where(sim_inner.c.sim_dist < 0.05)).one_or_none()
                    if same is not None:
                        add_to_group(obj.id, same[0],
                                f'names similar: {1 - same[3]:.4f} '
                                f'for {same[1]} ({same[2]}) vs {obj.name} ({obj.name_idx})')
                        return

                if not added_to_group:
                    # None of above matches worked. This object is in its own
                    # group
                    add_to_group(obj.id, obj.id, 'no match')
                return
        except sa.exc.IntegrityError:
            # Try again -- lock free, basically
            time.sleep(random.random())
        except:
            raise ValueError(f'While processing {entity_id}')

