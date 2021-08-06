
from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.db.temptable import temp_table_from_model
import sqlalchemy as sa

Base = sa.orm.declarative_base()
class EntitySearch(Base):
    __tablename__ = 'EntitySearch'

    id: int = sa.Column(sa.Integer, primary_key=True)
    name: str = sa.Column(sa.String, nullable=False)
    name_idx: str = sa.Column(sa.String, sa.Computed('pg_temp.name_fn(name)'))
    __table_args__ = (
        sa.Index('idx_entitysearch', 'name_idx',
            postgresql_ops={'name_idx': 'gin_trgm_ops'},
            postgresql_using='gin'),
    )

if __name__ == '__main__':
    #import logging
    #logging.basicConfig()
    #logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)
    with get_session() as sess:
        # Ensure fuzzy name matching and trigrams
        sess.execute(sa.text(r'''create extension if not exists fuzzystrmatch;'''))
        sess.execute(sa.text(r'''create extension if not exists pg_trgm;'''))
        # Must upload function first; use pg_temp for temporary
        sess.execute(sa.text(r'''create function pg_temp.name_fn(text) returns text as $$
                select metaphone($1, 20)
                $$ language sql immutable'''))

        ETable = temp_table_from_model(sess, EntitySearch)
        sess.execute(sa.insert(ETable).from_select(['id', 'name'],
                sa.select(sch.Entity.id, sch.Entity.attrs['name'].astext).where(
                    # Some entities have huge names... I think the way alembic
                    # makes the db differs from this method, where the default
                    # varchar length is 255.
                    (sch.Entity.type == 'person') & (sa.func.length(sch.Entity.name) < 255))
                ))

        # Automatically indexed!
        name_fn = sa.func.pg_temp.name_fn
        name = 'guido van roosum'
        thresh = 0.1
        explain_analyze = lambda sess, stmt: sess.execute('EXPLAIN ANALYZE ' + str(
                stmt.compile(compile_kwargs={'literal_binds': True})))

        cur = explain_analyze(sess, sa.select(ETable).where(
                sa.func.similarity(ETable.name_idx, name_fn(name)) > thresh))
        for row in cur:
            print(row[0])

        sim = sa.func.similarity(ETable.name_idx, name_fn(name))
        cur = sess.execute(
                sa.select(ETable, sim).where(sim > thresh)
                    .order_by(sim.desc())
                    .limit(20))
        for row in cur:
            r = row[0]
            print(f'{r.name} -- {row[1]}')

