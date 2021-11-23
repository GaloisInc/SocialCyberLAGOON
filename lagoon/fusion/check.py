"""Streamlit script for checking fusion results.
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import collections
import sqlalchemy as sa
import streamlit as st

def main():
    with get_session() as sess:
        ents = (sess.query(sch.EntityFusion.id_lowest, sa.func.count())
                .group_by(sch.EntityFusion.id_lowest)
                .order_by(sa.func.count().desc())
                .limit(25)
                .all())

        for ent_id, ent_count in ents:
            st.header(f'{ent_id} - {ent_count} underlying entities')

            by_comment = collections.defaultdict(list)
            for obj, comment in (
                    sess.query(sch.Entity, sch.EntityFusion.comment)
                    .join(sch.EntityFusion,
                        (sch.EntityFusion.id_other == sch.Entity.id)
                        & (sch.EntityFusion.id_lowest == ent_id))
                    .order_by(sch.EntityFusion.comment, sch.Entity.name)
                    .all()):
                by_comment[comment].append(obj)

            for comment, entity_list in sorted(by_comment.items(),
                    key=lambda m: -len(m[1])):
                st.subheader(f'{comment} -- {len(entity_list)} entities')
                for obj in entity_list:
                    with st.expander(f'{obj.name} -- {obj.type}'):
                        st.write(obj.attrs)

        # Don't change anything
        sess.rollback()


if __name__ == '__main__':
    main()

