"""Entity fusion program.
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
from lagoon.db.temptable import (temp_table_from_model, multi_session_dropper,
        multi_session_remap)

import multiprocessing
import random
import sqlalchemy as sa
import time
import tqdm

Base = sa.orm.declarative_base()
class EntitySearch(Base):
    __tablename__ = 'EntitySearch_fusion'

    id: int = sa.Column(sa.Integer, primary_key=True)
    name: str = sa.Column(sa.String(65534), index=True)
    email: str = sa.Column(sa.String(65534), index=True)

    name_idx: str = sa.Column(sa.String, sa.Computed('fusion_name_munge(name)'))

    __table_args__ = (
        sa.Index('idx_entitysearch_zz', 'name_idx',
            postgresql_ops={'name_idx': 'gin_trgm_ops'},
            postgresql_using='gin'),
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
        sess.execute(sa.text(r'''create or replace function fusion_name_munge(text) returns text as $$
                select metaphone($1, 20)
                $$ language sql immutable'''))

        ETable = temp_table_from_model(sess, EntitySearch, multi_session=True)

    with multi_session_dropper(get_session, ETable):
        with get_session() as sess:
            sess.execute(sa.insert(ETable).from_select(['id', 'name', 'email'],
                    sa.select(sch.Entity.id,
                        sa.func.lower(sch.Entity.attrs['name'].astext),
                        sa.func.lower(sch.Entity.attrs['email'].astext))
                    .where(
                        (sch.Entity.type == 'person') & (sa.func.length(sch.Entity.name) < 1000))
                    ))

            all_obj_ids = [o[0] for o in sess.execute(sa.select(ETable.id))]

            # Purge previous fusion results
            sess.execute(sa.delete(sch.EntityFusion))

            # Anything that's not a person is exempt from fusion
            sess.execute(sa.insert(sch.EntityFusion).from_select(['id_lowest', 'id_other'],
                    sa.select(sch.Entity.id, sch.Entity.id.label('id2'))
                    .where(sch.Entity.id.not_in(sa.select(ETable.id)))
                    ))

        # Merge every entity
        list(tqdm.tqdm(p.imap(_fuse_entity, all_obj_ids), total=len(all_obj_ids)))


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

                # Same email?
                same = sess.execute(sa.select(ET).where(
                        (ET.email == obj.email)
                        & (ET.id != obj.id)
                        ).order_by(ET.id).limit(1)).scalar()
                if same is not None:
                    add_to_group(obj.id, same.id, f'email match: {obj.email}')

                # Same name? Don't allow for short names
                if ' ' in obj.name and len(obj.name) > 5:
                    same = sess.execute(sa.select(ET).where(
                            (ET.name == obj.name)
                            & (ET.id != obj.id)
                            ).order_by(ET.id).limit(1)).scalar()
                    if same is not None:
                        add_to_group(obj.id, same.id, f'name match: {obj.name}')

                    sim = sa.func.similarity(ET.name_idx, obj.name_idx)
                    same = sess.execute(sa.select(ET, ET.name, ET.name_idx, sim).where(
                            (ET.id != obj.id)
                            & (sim > 0.95))
                            .order_by(ET.id).limit(1)).one_or_none()
                    if same is not None:
                        add_to_group(obj.id, same[0].id,
                                f'names similar: {same[3]:.4f} '
                                f'for {same[1]} ({same[2]}) vs {obj.name} ({obj.name_idx})')

                if not added_to_group:
                    # None of above matches worked. This object is in its own
                    # group
                    add_to_group(obj.id, obj.id, 'no match')
                return
        except sa.exc.IntegrityError:
            # Try again -- lock free, basically
            time.sleep(random.random())

