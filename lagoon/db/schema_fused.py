"""Provides wrappers around :mod:`lagoon.db.schema` classes which give a
read-only view of fused data.

See `lagoon/fusion/fused_cache.py` for information on how these are generated.
"""

from . import schema as sch

import dataclasses
import datetime
import sqlalchemy as sa
from typing import Any, Dict

# union_all expensive in Postgres; do it as a lateral w/ match criteria
# Note - this query ALWAYS returns at least 1 row for any AttrsBase-derived
# class.
def _attrs_sources_lateral_fn(ent_id_match):
    CA = sa.orm.aliased(sch.ComputedAttrs)
    AB = sa.orm.aliased(sch.AttrsBase)

    return (
            sa.select(CA.obj_id.label('ent_id'), CA.id)
                .where(CA.obj_id == ent_id_match)
            .union_all(
                sa.select(AB.id.label('ent_id'), AB.id)
                    .where(AB.id == ent_id_match)
            )
    ).lateral()


@dataclasses.dataclass
class FusedEntity(sch.Base, sch.DataClassMixin):
    __tablename__ = 'cache_fused_entity'
    __repr__ = sch.Entity.__repr__

    id: int = sa.Column(sa.Integer, primary_key=True)
    name: str = sa.Column(sa.String, nullable=False)
    type: sch.EntityTypeEnum = sa.Column(sch.DbEnum(sch.EntityTypeEnum),
            nullable=False)
    attrs: Dict[str, Any] = sa.Column(sch.DbJson(), nullable=False,
            default=lambda: {})

    fusions = sa.orm.relationship('EntityFusion',
            primaryjoin='foreign(EntityFusion.id_lowest) == FusedEntity.id',
            viewonly=True)

    @property
    def attrs_sources(self):
        """Returns a list which has, in order of reverse importance, each object
        whose `attrs` field is merged into this `FusedEntity`'s.
        """
        sess = sa.orm.object_session(self)

        ids = _attrs_sources_lateral_fn(sch.EntityFusion.id_other)
        res_type = sa.orm.with_polymorphic(sch.AttrsBase,
                [sch.Entity, sch.ComputedAttrs])
        subq = (
                sess.query(res_type)
                .select_from(sch.EntityFusion)
                .where(sch.EntityFusion.id_lowest == self.id)
                .join(ids, sa.true())
                .join(res_type, res_type.id == ids.c.id)
                )
        return subq.order_by(sch.AttrsBase.id.asc()).all()


    def obs_hops(self, k, time_min=None, time_max=None, *,
            sample_growth=0):
        """Returns an array of all observations (which include links to entities)
        within `k` hops of this entity. Optionally, those observations may be
        bounded by time range.

        Args:
            sample_growth (int): If non-zero, then each hop allows merging on
                    a maximum of `sample_growth` new observations per entity.
        """

        limit = []
        if time_min is not None:
            if isinstance(time_min, float):
                time_min = datetime.datetime.fromtimestamp(time_min)
            limit.append(FusedObservation.time >= time_min)
        if time_max is not None:
            if isinstance(time_max, float):
                time_max = datetime.datetime.fromtimestamp(time_max)
            limit.append(FusedObservation.time < time_max)

        # Use -1 as it will not exist in the db
        q = sa.select(sa.literal(-1).label('id'))
        q_prev = sa.select(sa.literal(-1).label('id_linked'))
        q_next = sa.select(sa.literal(self.id).label('id_linked'))
        for i in range(k):
            q_new = (
                    sa.select(q_next.c.id_linked, FusedObservation.id, FusedObservation.src_id)
                    .where(FusedObservation.dst_id == q_next.c.id_linked, *limit)
                    .union(
                        sa.select(q_next.c.id_linked, FusedObservation.id, FusedObservation.dst_id)
                        .where(FusedObservation.src_id == q_next.c.id_linked, *limit)))

            if sample_growth:
                # Limit by random sort
                q_new = q_new.subquery()
                q_new = sa.select(q_new, sa.func.row_number().over(
                    partition_by=q_new.c.id_linked,
                    order_by=sa.func.random()).label('row_number'))
                q_new = sa.select(q_new.c.id, q_new.c.src_id).where(
                        q_new.c.row_number <= sample_growth)
                # Note -- cte() VERY important. Otherwise, the `row_number`
                # by random clause gets executed multiple times, one for each
                # hop.
                q_new = q_new.cte()

            # Add newfound edges
            q = sa.union(q, sa.select(q_new.c.id))
            # Mark entities explored
            q_prev = sa.union_all(q_prev, sa.select(q_next.c.id_linked))
            # Note entities which need to be explored in the next loop
            q_next = sa.select(q_new.c.src_id.label('id_linked')).where(
                    ~sa.exists().where(q_new.c.src_id == q_prev.c.id_linked))

        sess = sa.orm.object_session(self)
        obj_query = sess.query(FusedObservation).where(
                FusedObservation.id == q.c.id)
        # Pull in the entities efficiently
        obj_query = obj_query.options(sa.orm.selectinload(FusedObservation.dst))
        obj_query = obj_query.options(sa.orm.selectinload(FusedObservation.src))
        r = obj_query.all()
        return r


@dataclasses.dataclass
class FusedObservation(sch.Base, sch.DataClassMixin):
    __tablename__ = 'cache_fused_observation'
    __repr__ = sch.Observation.__repr__

    id: int = sa.Column(sa.Integer, primary_key=True)
    batch_id: int = sa.Column(sa.Integer, nullable=False, index=True)
    type: sch.ObservationTypeEnum = sa.Column(sch.DbEnum(sch.ObservationTypeEnum),
            nullable=False)
    time: datetime.datetime = sa.Column(sa.DateTime, nullable=False)
    attrs: Dict[str, Any] = sa.Column(sch.DbJson(), nullable=False,
            default=lambda: {})
    dst_id: int = sa.Column(sa.Integer, sa.ForeignKey('cache_fused_entity.id'),
            nullable=False)
    # https://docs.sqlalchemy.org/en/14/orm/query.html#sqlalchemy.orm.with_parent
    # Waiting on https://github.com/sqlalchemy/sqlalchemy/issues/6855
    # Usage:
    # session.execute(sa.select(sch.FusedObservation).where(
    #     sa.orm.with_parent(entity, sch.FusedEntity.obs_as_src))).all()
    dst = sa.orm.relationship('FusedEntity',
            backref=sa.orm.backref('obs_as_dst', lazy='raise'),
            primaryjoin='foreign(FusedObservation.dst_id) == FusedEntity.id',
            viewonly=True)
    src_id: int = sa.Column(sa.Integer, sa.ForeignKey('cache_fused_entity.id'),
            nullable=False)
    src = sa.orm.relationship('FusedEntity',
            backref=sa.orm.backref('obs_as_src', lazy='raise'),
            primaryjoin='foreign(FusedObservation.src_id) == FusedEntity.id',
            viewonly=True)


    __table_args__ = (
            sa.Index('idx_fused_observation_dst_by_timestamp', 'dst_id', 'time'),
            sa.Index('idx_fused_observation_src_by_timestamp', 'src_id', 'time'),
    )

    @property
    def attrs_sources(self):
        """Returns a list which has, in order of reverse importance, each object
        whose `attrs` field is merged into this `FusedObservation`'s.
        """
        sess = sa.orm.object_session(self)

        res_type = sa.orm.with_polymorphic(sch.AttrsBase,
                [sch.Observation, sch.ComputedAttrs])
        ids = _attrs_sources_lateral_fn(self.id)
        subq = (
                sess.query(res_type)
                .select_from(ids)
                .join(res_type, res_type.id == ids.c.id)
                )
        return subq.order_by(sch.AttrsBase.id.asc()).all()

