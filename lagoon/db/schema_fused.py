"""Provides wrappers around :mod:`lagoon.db.schema` classes which give a
read-only view of fused data.
"""

from . import schema as sch

import dataclasses
import datetime
import sqlalchemy as sa
from typing import Any, Dict

# Limit to only those which are id_lowest, for safety
_entity_is_lowest = sa.select(sch.EntityFusion).where(
        sch.EntityFusion.id_lowest == sch.Entity.id).exists()
_entity_query = sa.select(sch.Entity).where(_entity_is_lowest).subquery()
@dataclasses.dataclass
class FusedEntity(sch.Base):
    __table__ = _entity_query
    __repr__ = sch.Entity.__repr__

    id: int
    batch_id: int
    name: str
    type: sch.EntityTypeEnum
    attrs: Dict[str, Any]

    fusions = sa.orm.relationship('EntityFusion',
            primaryjoin='EntityFusion.id_lowest == FusedEntity.id',
            viewonly=True)

    def obs_hops(self, k, time_min=None, time_max=None):
        """Returns an array of all observations (which include links to entities)
        within `k` hops of this entity. Optionally, those observations may be
        bounded by time range.
        """

        # Use -1 as it will not exist in the DB
        q = sa.select(sa.literal(-1).label('id'), sa.literal(self.id).label('id_linked'))
        limit = []
        if time_min is not None:
            limit.append(FusedObservation.time >= time_min)
        if time_max is not None:
            limit.append(FusedObservation.time <= time_max)
        for i in range(k):
            q = sa.union(q,
                    sa.select(FusedObservation.id, FusedObservation.src_id).where(
                        FusedObservation.dst_id == q.c.id_linked, *limit)
                    .union(
                        sa.select(FusedObservation.id, FusedObservation.dst_id).where(
                            FusedObservation.src_id == q.c.id_linked, *limit))
                    )
        obj_query = sa.select(FusedObservation).where(
                FusedObservation.id == q.c.id)
        # Pull in the entities efficiently
        obj_query = obj_query.options(sa.orm.selectinload(FusedObservation.dst))
        obj_query = obj_query.options(sa.orm.selectinload(FusedObservation.src))
        sess = sa.orm.object_session(self)
        return [o[0] for o in sess.execute(obj_query)]


# Rewrite FusedObservation s.t. the dst_id and src_id fields are replaced
# correctly with fused versions.
_O = sch.Observation
_F1 = sa.orm.aliased(sch.EntityFusion)
_F2 = sa.orm.aliased(sch.EntityFusion)
_obs_query = (
        sa.select(_O.id, _O.batch_id, _O.type, _O.value, _O.time,
            _O.attrs, _F1.id_lowest.label('src_id'), _F2.id_lowest.label('dst_id'))
        .select_from(_O)
        .join(_F1, _O.src_id == _F1.id_other)
        .join(_F2, _O.dst_id == _F2.id_other)
        ).subquery()
@dataclasses.dataclass
class FusedObservation(sch.Base):
    __table__ = _obs_query
    __repr__ = sch.Observation.__repr__

    id: int
    batch_id: int
    type: sch.ObservationTypeEnum
    value: float
    time: datetime.datetime
    attrs: Dict[str, Any]
    dst_id: int
    src_id: int

    # https://docs.sqlalchemy.org/en/14/orm/query.html#sqlalchemy.orm.with_parent
    # Waiting on https://github.com/sqlalchemy/sqlalchemy/issues/6855
    # Usage:
    # session.execute(sa.select(sch.FusedObservation).where(
    #     sa.orm.with_parent(entity, sch.FusedEntity.obs_as_src))).all()
    dst = sa.orm.relationship('FusedEntity',
            backref=sa.orm.backref('obs_as_dst', lazy='raise'),
            primaryjoin='foreign(FusedObservation.dst_id) == FusedEntity.id',
            viewonly=True,
            )
    src = sa.orm.relationship('FusedEntity',
            backref=sa.orm.backref('obs_as_src', lazy='raise'),
            primaryjoin='foreign(FusedObservation.src_id) == FusedEntity.id',
            viewonly=True,
            )

