
from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.db.temptable import temp_table_from_model, multi_session_dropper

import arrow
import datetime
import sqlalchemy as sa
import streamlit as st

st.set_page_config(layout='wide')

cache_dec = lambda: st.cache(allow_output_mutation=True,
        hash_funcs={'sqlalchemy.orm.session.Session': lambda x: 'hi',
            'sqlalchemy.orm.attributes.InstrumentedAttribute': lambda x: 'hi'})

Base = sa.orm.declarative_base()
class FileDetails(Base):
    '''For keeping statistics'''
    __tablename__ = 'FileDetails_temp'
    row_num: int = sa.Column(sa.Integer, primary_key=True)
    fname: str = sa.Column(sa.String(), index=True)
    touched: datetime.datetime = sa.Column(sa.DateTime, index=True)
    who: int = sa.Column(sa.Integer(), index=True)


def setup_table():
    with get_session() as sess:
        FTable = temp_table_from_model(sess, FileDetails, multi_session='streamlit')
        if st.button('Delete table?'):
            with multi_session_dropper(get_session, FTable):
                pass
            st.write('Deleted')
            return

        if st.button('Regenerate table?'):
            files = (sess.query(sch.FusedEntity)
                    .where(sch.FusedEntity.type == sch.EntityTypeEnum.file))
            fcount = files.count()
            fprogress = st.progress(0.)
            for ri, r in enumerate(files.yield_per(128)):
                fprogress.progress(ri / fcount)

                o1 = sa.orm.aliased(sch.FusedObservation)
                e2 = sa.orm.aliased(sch.FusedEntity)
                o2 = sa.orm.aliased(sch.FusedObservation)
                commits = (sess.query(o1.time, o2.src_id)
                        .select_from(o1)
                        .where(o1.dst_id == r.id)
                        .where(o1.type == sch.ObservationTypeEnum.modified)
                        .join(e2, o1.src)
                        .where(e2.type == sch.EntityTypeEnum.git_commit)
                        .join(o2, o2.dst_id == e2.id)
                        # To make "count == 1" valid, only look at created
                        # for now.
                        #.where(o2.type.in_([sch.ObservationTypeEnum.created,
                        #    sch.ObservationTypeEnum.committed]))
                        .where(o2.type == sch.ObservationTypeEnum.created)
                        ).all()
                for ctime, cwho in commits:
                    sess.add(FTable(fname=r.name, touched=ctime, who=cwho))
                sess.flush()

    return FTable

FTable = setup_table()
with get_session() as sess:
    st.write(f'Table has {sess.query(FTable).count()} rows')

    time_clause = (FTable.touched > arrow.get().shift(years=-3).datetime)

    # Look at:
    # 1. Files that have been touched by few unique maintainers within the last
    #    X years, where
    f1 = (sess.query(FTable.fname,
                sa.func.count(FTable.who.distinct()).label('count_unique'),
                sa.func.count().label('count_touches'))
            .select_from(FTable)
            .where(time_clause)
            .group_by(FTable.fname)
            ).cte()

    # 2. Those maintainers haven't touched other parts of the code recently,
    #    where
    cte_who = (
            sess.query(FTable.who, sa.func.count().label('count'))
            .where(time_clause)
            .group_by(FTable.who)
            ).cte()

    # 3. They have touched a specific file exactly one time in all time, and it
    #    was the recent attempt.
    cte_one = (
            sess.query(FTable.who, FTable.fname,
                sa.func.count().label('count_alltime'),
                sa.func.sum(sa.case([(time_clause, 1)])).label('count_recent'))
            .group_by(FTable.fname, FTable.who)
            ).subquery()
    cte_one = (
            sess.query(cte_one.c.who, cte_one.c.fname)
            .where(cte_one.c.count_alltime == 1)
            .where(cte_one.c.count_recent == 1)
            ).cte()

    query_str = st.text_input('Query for specific file? leave empty for all files')
    st.write('Sorted by (files with fewest unique maintainers, maintainers with fewest commits)')
    st.write('Rows of (User ID, number of user commits, file they touched one time, unique contributors to file, commits against file)')
    suspect = (
            sess.query(cte_one.c.who, cte_who.c.count,
                f1.c.fname, f1.c.count_unique, f1.c.count_touches)
            .select_from(f1)
            .where(f1.c.fname.op('~*')(
                r'\.(c|py)$'
                if not query_str.strip()
                else query_str.strip()
                ))
            .join(cte_one, cte_one.c.fname == f1.c.fname)
            .join(cte_who, cte_who.c.who == cte_one.c.who)
            #.where(f1.c.count_unique > 1)
            .where(cte_who.c.count == 1)
            .order_by(f1.c.count_unique, cte_who.c.count)
            )
    st.write(suspect.limit(20).all())

