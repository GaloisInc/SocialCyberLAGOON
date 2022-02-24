from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import altair as alt
import arrow
import contextlib
import pandas as pd
import streamlit as st
import sqlalchemy as sa
import time

# Fix for sqlalchemy
st_cache = lambda: st.cache(allow_output_mutation=True,
        hash_funcs={'sqlalchemy.orm.session.Session': lambda x: 'hi',
            'sqlalchemy.orm.attributes.InstrumentedAttribute': lambda x: 'hi',
            'sqlalchemy.orm.util.AliasedClass': lambda x: 'hi'})

@contextlib.contextmanager
def timeit(name):
    a = time.monotonic()
    yield
    b = time.monotonic()
    st.text(f'{name} -- {b - a:.3f}s')


p_table = sa.orm.aliased(sch.FusedEntity)

o1 = sa.orm.aliased(sch.FusedObservation) #p connects to c1 through o1
c1 = sa.orm.aliased(sch.FusedEntity) #commits from p
o2 = sa.orm.aliased(sch.FusedObservation) #c1 connects to f through o2
f = sa.orm.aliased(sch.FusedEntity) #files
o3 = sa.orm.aliased(sch.FusedObservation) # f connects to c2 through o3
c2 = sa.orm.aliased(sch.FusedEntity) #commits from ps neighbors
o4 =  sa.orm.aliased(sch.FusedObservation) # c2 connects to neighbor through o4
neighbor = sa.orm.aliased(sch.FusedEntity)

person_id = st.text_input('Person ID', value=130975)
person_id = int(person_id)

with get_session() as sess:
    p = sess.get(sch.FusedEntity, person_id)  # 130975 = Guido, 130909 = Ethan Furman
    sess.expunge(p)

with get_session() as sess:
    person_to_file = st.checkbox('Person to file', value=False)
    if person_to_file:
        with timeit('Our query'):
            neighbors = (sess.query(f)
                .select_from(p_table)
                .where(p.id==p_table.id)
                .join(o1, o1.src_id==p_table.id)
                .join(c1, o1.dst_id==c1.id )
                .where(c1.type == sch.EntityTypeEnum.git_commit)
                .join(o2, o2.src_id==c1.id)
                .join(f, o2.dst_id==f.id )
                .where(f.type == sch.EntityTypeEnum.file)
                ).distinct().count()

        with timeit('Baseline query, obs_hops only'):
            pp = sess.merge(p)
            neigh = pp.obs_hops(2)

    person_to_person = st.checkbox('Person to person', value=False)
    if person_to_person:
        year = st.text_input('Year', value=2014)
        year = int(year)

        @st_cache()
        def year_neighbors(p_id, year):
            t1 = arrow.get(f'{year}-01-01').datetime
            t2 = arrow.get(f'{year+1}-01-01').datetime
            neighbors = (sess.query(neighbor)
                .select_from(p_table)
                .where(p_id==p_table.id)
                .join(o1, o1.src_id==p_table.id)
                .where((o1.time >= t1) & (o1.time < t2))
                .join(c1, o1.dst_id==c1.id )
                .where(c1.type == sch.EntityTypeEnum.git_commit)
                .join(o2, o2.src_id==c1.id)
                .where((o2.time >= t1) & (o2.time < t2))
                .join(f, o2.dst_id==f.id )
                .where(f.type == sch.EntityTypeEnum.file)
                .join(o3, o3.dst_id==f.id)
                .where((o3.time >= t1) & (o3.time < t2))
                .join(c2, o3.src_id==c2.id )
                .where(c2.type == sch.EntityTypeEnum.git_commit)
                .join(o4, o4.dst_id==c2.id)
                .where((o4.time >= t1) & (o4.time < t2))
                .join(neighbor, o4.src_id==neighbor.id )
                .where(neighbor.type == sch.EntityTypeEnum.person)
                ).distinct().count()
            return neighbors

        st.write("Clear cache (with 'c') before re-running timer")
        with timeit('Our query'):
            year_neighbors(p.id, year)

        st.write('Generating plot, to be fancy')
        data = {'year': range(1990, 2022)}
        data['neighbors'] = [year_neighbors(p.id, y) for y in data['year']]
        chart = alt.Chart(pd.DataFrame(data)).mark_line().encode(
                x='year:T', y='neighbors:Q')
        st.altair_chart(chart, use_container_width=True)

    sess.rollback()

