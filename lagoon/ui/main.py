
from .plugin import Plugin

import lagoon.db.connection
import lagoon.db.schema as sch

import arrow
import collections
import dataclasses
import datetime
import importlib
import inspect
import os
import re
import sqlalchemy as sa
from typing import List
import vuespa

# Initialized on UI startup
plugins_by_cap = collections.defaultdict(list)
def _load_plugins(plugins):
    """Given some list of strings like `"module_path,class_name:argument"`, load
    the specified plugins.
    """
    for p in plugins:
        pfile, pclass, parg = p, None, None
        if ':' in pfile:
            pfile, parg = pfile.split(':', 1)
        if ',' in pfile:
            pfile, pclass = pfile.split(',', 1)

        if '/' in pfile or pfile.endswith('.py'):
            # Assume this is an OS path. Try to convert it to a python module
            # name.
            assert not pfile.startswith('/'), 'Must be a normal python import'
            if pfile.endswith('.py'):
                pfile = pfile[:-3]

            pfile = re.split(r'[/.]', pfile)
        else:
            pfile = pfile.split('.')

        pmod = importlib.import_module('.'.join(pfile))
        if pclass is None:
            for k, v in pmod.__dict__.items():
                if inspect.isclass(v) and issubclass(v, Plugin) and v is not Plugin:
                    if pclass is not None:
                        raise ValueError(f'More than 1 Plugin in {pfile}; '
                                f'use e.g. `module.name,class_name` to '
                                f'specify a single plugin class.')
                    pclass = k
        if pclass is None:
            raise ValueError(f'No plugin found? {pfile}')

        cls = getattr(pmod, pclass)
        inst = cls(parg)
        for cap in inst.plugin_caps():
            plugins_by_cap[cap].append(inst)


class DbEncoder:
    """Encode objects from sqlalchemy as JSON-compatible.
    """
    @classmethod
    def encode(cls, o):
        if o is None:
            return o

        try:
            d = dataclasses.asdict(o)
        except:
            raise ValueError(type(o))

        # For UI, add extra properties
        if isinstance(o, sch.FusedEntity):
            # computed_badwords_*
            bw = {'body': 0, 'quote': 0, 'sign': 0}
            for k, v in d['attrs'].items():
                if k.startswith('computed_badwords_'):
                    bwk = k.split('_')[2]
                    bw[bwk] += v
            for k, v in bw.items():
                d[f'ui_computed_badwords_{k}'] = v

        if isinstance(o, (sch.Entity, sch.Observation, sch.FusedEntity,
                sch.FusedObservation)):
            d['type'] = d['type'].value
        else:
            # Assume other object types are JSON-compatible.
            pass

        if isinstance(o, (sch.Entity, sch.FusedEntity)):
            for pi, p in enumerate(plugins_by_cap['plugin_details_entity']):
                p_name = p.plugin_name()
                d['attrs'][p_name] = {'$plugin': pi}

        if isinstance(o, sch.FusedEntity) and 'fusions' in o.__dict__:
            d['fusions'] = [DbEncoder.encode(f) for f in o.fusions]
            for ff, f in zip(d['fusions'], o.fusions):
                if hasattr(f, 'ui_batch_other'):
                    ff['ui_batch_other'] = DbEncoder.encode(f.ui_batch_other)

        d = {k:
                arrow.get(v).timestamp() if isinstance(v, datetime.datetime)
                else v for k, v in d.items()}
        d['repr'] = repr(o)
        return d



class Client(vuespa.Client):
    async def vuespa_on_open(self):
        pass
    async def vuespa_on_close(self):
        pass
    async def api_entity_random(self):
        async with lagoon.db.connection.get_session_async() as sess:
            q = await sess.execute(sa.select(sch.FusedEntity).limit(1))
            o = q.scalar()
            return DbEncoder.encode(o)
    async def api_entity_get(self, i):
        async with lagoon.db.connection.get_session_async() as sess:
            def o_sync(session):
                q = session.execute(sa.select(sch.FusedEntity).filter(sch.FusedEntity.id == i))
                o = q.scalar()
                # Load fusions
                o.fusions
                # Stock batch information from fusions
                qq = (
                        session.query(sch.Entity.id, sch.Batch)
                        .select_from(sch.Entity)
                        .where(sch.Entity.id.in_([f.id_other for f in o.fusions]))
                        .join(sch.Batch, sch.Entity.batch)
                        ).all()
                qq = {k: v for k, v in qq}
                for f in o.fusions:
                    f.ui_batch_other = qq[f.id_other]
                return o
            r = await sess.run_sync(o_sync)
            return DbEncoder.encode(r)
    async def api_entity_get_obs(self, i, t_start, t_end):
        """Returns [as src, as dst]"""
        t_start = arrow.get(t_start).datetime.replace(tzinfo=None)
        t_end = arrow.get(t_end).datetime.replace(tzinfo=None)
        async with lagoon.db.connection.get_session_async() as sess:
            def o_sync(session):
                q = session.execute(sa.select(sch.FusedEntity).filter(sch.FusedEntity.id == i))
                o = q.scalar()
                query = sa.and_(sch.FusedObservation.time >= t_start,
                        sch.FusedObservation.time <= t_end)

                q_src = (
                        sa.orm.dynamic.Query(sch.FusedObservation, session=session)
                        .where(sa.orm.with_parent(o, sch.FusedEntity.obs_as_src))
                        .filter(query)
                        .all())
                q_dst = (
                        sa.orm.dynamic.Query(sch.FusedObservation, session=session)
                        .where(sa.orm.with_parent(o, sch.FusedEntity.obs_as_dst))
                        .filter(query)
                        .all())
                return q_src, q_dst
            src, dst = await sess.run_sync(o_sync)
            return [DbEncoder.encode(o) for o in src + dst]
    async def api_entity_obs_adjacent(self, i, t_start, t_end):
        t_start = arrow.get(t_start).datetime.replace(tzinfo=None)
        t_end = arrow.get(t_end).datetime.replace(tzinfo=None)
        async with lagoon.db.connection.get_session_async() as sess:
            O = sch.FusedObservation
            E = sch.FusedEntity
            def o_sync(session):
                o = session.execute(sa.select(E).filter(E.id == i)).scalar()
                q = (O.time < t_start)
                obs_base = sa.orm.dynamic.Query(sch.FusedObservation, session=session)
                obs_as_src = obs_base.with_parent(o, sch.FusedEntity.obs_as_src)
                obs_as_dst = obs_base.with_parent(o, sch.FusedEntity.obs_as_dst)
                before = [
                        obs_as_src.filter(q).order_by(O.time.desc()).limit(1).scalar(),
                        obs_as_dst.filter(q).order_by(O.time.desc()).limit(1).scalar()]
                before = [b for b in before if b is not None]
                if before:
                    before = sorted(before, key=lambda m: -arrow.get(m.time).timestamp())[0]
                else:
                    before = None
                q = (O.time > t_end)
                after = [
                        obs_as_src.filter(q).order_by(O.time).limit(1).scalar(),
                        obs_as_dst.filter(q).order_by(O.time).limit(1).scalar()]
                after = [b for b in after if b is not None]
                if after:
                    after = sorted(after, key=lambda m: arrow.get(m.time).timestamp())[0]
                else:
                    after = None
                return before, after
            before, after = await sess.run_sync(o_sync)
            return [DbEncoder.encode(o) for o in [before, after]]
    async def api_entity_search(self, s):
        """Find entities with name matching the regex s"""
        async with lagoon.db.connection.get_session_async() as sess:
            limit = 10
            # `cached_names` is newline-separated.... so replace the
            # anchor appropriately.
            import sre_parse

            s_orig = s
            s_p = sre_parse.parse(s)
            def unparse_and_fix(ss):
                r = []
                for k, v in ss:
                    if k == sre_parse.ANY:
                        if v is not None:
                            raise NotImplementedError(f'Value {repr(v)}')
                        # Replicating 'm' flag
                        r.append(r'[^\n]')
                    elif k == sre_parse.LITERAL:
                        v = chr(v)
                        if v in '\\.^$()[]{}?*+|/-':
                            r.append('\\')
                        r.append(v)
                    elif k == sre_parse.MAX_REPEAT:
                        r.append('(')
                        r.append(unparse_and_fix(v[2]))
                        r.append(')')

                        if v[0] == 0:
                            if v[1] == 1:
                                r.append('?')
                            elif v[1] == sre_parse.MAXREPEAT:
                                r.append('*')
                            else:
                                raise NotImplementedError(f"Max repeat count: {v[1]}")
                        elif v[0] == 1:
                            if v[1] != sre_parse.MAXREPEAT:
                                raise NotImplementedError("Max repeat count")
                            r.append('+')
                        else:
                            raise NotImplementedError(v[0])
                    elif k == sre_parse.AT:
                        subs = {
                                # The one thing to change
                                sre_parse.AT_BEGINNING: r'(^|\n)',
                                sre_parse.AT_END: '$',
                        }
                        v_r = subs.get(v)
                        if v_r is None:
                            raise NotImplementedError(v)
                        r.append(v_r)
                    elif k == sre_parse.SUBPATTERN:
                        r.append('(')
                        r.append(unparse_and_fix(v[3]))
                        r.append(')')
                    elif k == sre_parse.BRANCH:
                        if v[0] is not None:
                            raise NotImplementedError(v)
                        r.append('(?:')
                        for vv_i, vv in enumerate(v[1]):
                            if vv_i != 0:
                                r.append('|')
                            r.append(unparse_and_fix(vv))
                        r.append(')')
                    else:
                        raise NotImplementedError(k)
                return ''.join(r)
            try:
                s = unparse_and_fix(s_p)
            except NotImplementedError as e:
                return [
                        {'value': None, 'label': f'Regex: {repr(s)}'},
                        {'value': None, 'label': f'Not implemented: {e}'}]

            q = await sess.execute(sa.select(sch.FusedEntity).filter(
                    # ~* is case-insensitive posix regex in postgres
                    sch.FusedEntity.cached_names.op('~*')(s)).limit(limit))
            q = [qq[0] for qq in q.all()]

            # See if there's an integer to search for in there
            ss = s_orig
            if ss.startswith('^'):
                ss = ss[1:]
            if ss.endswith('.*'):
                ss = ss[:-2]
            try:
                maybe_id = int(ss)
            except ValueError:
                pass
            else:
                q_id = await sess.execute(sa.select(sch.FusedEntity).filter(
                        sch.FusedEntity.id == maybe_id))
                q = [qq[0] for qq in q_id.all()] + q

            # Convert to JS object
            r = [{'value': o.id, 'label': repr(o)} for o in q]
            return r
    async def api_entity_search_reverse(self, i):
        """Returns entity label for given id.
        """
        async with lagoon.db.connection.get_session_async() as sess:
            q = await sess.execute(sa.select(sch.FusedEntity).filter(
                    sch.FusedEntity.id == i))
            return repr(q.scalar())
    async def api_plugin_details_entity(self, p, i):
        """Returns the information for the given entity from a plugin.

        Args:
            p: plugin index
            i: entity ID
        """
        plugin = plugins_by_cap['plugin_details_entity'][p]
        async with lagoon.db.connection.get_session_async() as sess:
            def o_sync(session):
                ent = session.query(sch.FusedEntity).where(
                        sch.FusedEntity.id == i).scalar()
                return plugin.plugin_details_entity(ent)
            result = await sess.run_sync(o_sync)
        return {'$html': result}


def main(plugins: List[str]):
    """Typically called via `lagoon_cli.py ui [options]`"""
    print('Loading plugins')
    _load_plugins(plugins)

    print('Starting UI')
    path = os.path.dirname(os.path.abspath(__file__))
    vuespa.VueSpa(os.path.join(path, 'vue-ui'), Client, port=8070).run()

