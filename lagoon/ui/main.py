
import lagoon.db.connection
import lagoon.db.schema as sch

import arrow
import dataclasses
import datetime
import os
import sqlalchemy as sa
import vuespa

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
        if isinstance(o, (sch.Entity, sch.Observation, sch.FusedEntity,
                sch.FusedObservation)):
            d['type'] = d['type'].value
        else:
            # Assume other object types are JSON-compatible.
            pass

        if isinstance(o, sch.FusedEntity) and 'fusions' in o.__dict__:
            d['fusions'] = [DbEncoder.encode(f) for f in o.fusions]

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
        """Find entities with name like s (replaces * with % for db)"""
        async with lagoon.db.connection.get_session_async() as sess:
            limit = 10
            q = await sess.execute(sa.select(sch.FusedEntity).filter(
                    # ~* is case-insensitive posix regex in postgres
                    sch.FusedEntity.name.op('~*')(s)).limit(limit))
            q = [qq[0] for qq in q.all()]

            # See if there's an integer to search for in there
            ss = s
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


def main():
    """Typically called via `lagoon_cli.py ui [options]`"""
    path = os.path.dirname(os.path.abspath(__file__))
    vuespa.VueSpa(os.path.join(path, 'vue-ui'), Client, port=8070).run()

