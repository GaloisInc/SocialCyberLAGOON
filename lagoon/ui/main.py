
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

        d = dataclasses.asdict(o)
        if isinstance(o, (sch.Entity, sch.Observation)):
            d['type'] = d['type'].value
        else:
            # Assume other object types are JSON-compatible.
            pass

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
            q = await sess.execute(sa.select(sch.Entity).limit(1))
            o = q.scalar()
            return DbEncoder.encode(o)
    async def api_entity_get(self, i):
        async with lagoon.db.connection.get_session_async() as sess:
            q = await sess.execute(sa.select(sch.Entity).filter(sch.Entity.id == i))
            o = q.scalar()
            return DbEncoder.encode(o)
    async def api_entity_get_obs(self, i, t_start, t_end):
        """Returns [as src, as dst]"""
        t_start = arrow.get(t_start).datetime.replace(tzinfo=None)
        t_end = arrow.get(t_end).datetime.replace(tzinfo=None)
        async with lagoon.db.connection.get_session_async() as sess:
            q = await sess.execute(sa.select(sch.Entity).filter(sch.Entity.id == i))
            o = q.scalar()
            def o_sync(session):
                query = sa.and_(sch.Observation.time >= t_start,
                        sch.Observation.time <= t_end)
                return o.obs_as_src.filter(query).all(), o.obs_as_dst.filter(query).all()
            src, dst = await sess.run_sync(o_sync)
            return [DbEncoder.encode(o) for o in src + dst]
    async def api_entity_obs_adjacent(self, i, t_start, t_end):
        t_start = arrow.get(t_start).datetime.replace(tzinfo=None)
        t_end = arrow.get(t_end).datetime.replace(tzinfo=None)
        async with lagoon.db.connection.get_session_async() as sess:
            O = sch.Observation
            E = sch.Entity
            o = (await sess.execute(sa.select(E).filter(E.id == i))).scalar()
            def o_sync(session):
                q = (O.time < t_start)
                before = [
                        o.obs_as_src.filter(q).order_by(O.time.desc()).limit(1).scalar(),
                        o.obs_as_dst.filter(q).order_by(O.time.desc()).limit(1).scalar()]
                before = [b for b in before if b is not None]
                if before:
                    before = sorted(before, key=lambda m: -arrow.get(m.time).timestamp())[0]
                else:
                    before = None
                q = (O.time > t_end)
                after = [
                        o.obs_as_src.filter(q).order_by(O.time).limit(1).scalar(),
                        o.obs_as_dst.filter(q).order_by(O.time).limit(1).scalar()]
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
            q = await sess.execute(sa.select(sch.Entity).filter(
                    # ~* is case-insensitive posix regex in postgres
                    sch.Entity.name.op('~*')(s)).limit(limit))
            q = [qq[0] for qq in q.all()]
            r = [{'value': o.id, 'label': repr(o)} for o in q]
            return r
    async def api_entity_search_reverse(self, i):
        """Returns entity label for given id.
        """
        async with lagoon.db.connection.get_session_async() as sess:
            q = await sess.execute(sa.select(sch.Entity).filter(
                    sch.Entity.id == i))
            return repr(q.scalar())


def main():
    """Typically called via `lagoon_cli.py ui [options]`"""
    path = os.path.dirname(os.path.abspath(__file__))
    vuespa.VueSpa(os.path.join(path, 'vue-ui'), Client, port=8070).run()

