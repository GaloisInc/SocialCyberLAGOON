"""Provides wrappers around :mod:`lagoon.db.schema` classes which give a
read-only view of fused data.
"""

from . import schema as sch

import dataclasses
import datetime
import sqlalchemy as sa
from typing import Any, Dict

###############################################################################
# UGH awful hack to work with SqlAlchemy + Postgres' proprietary 'ORDER BY'
# extension within jsonb_object_agg
class _ArgumentOrderBy(sa.sql.functions.ColumnElement):
    __visit_name__ = "argumentorderby"
    _traverse_internals = [
            ("element", sa.sql.functions.InternalTraversal.dp_clauseelement),
            ("order_by", sa.sql.functions.InternalTraversal.dp_clauseelement),
    ]
    def __init__(self, element, *order_by):
        super().__init__()
        assert order_by, 'order_by is required'
        self.element = element
        self.order_by = sa.sql.functions.ClauseList(
                *sa.util.to_list(order_by),
                _literal_as_text_role=sa.sql.roles.ByOfRole)
def _visit_argumentorderby(self, argument, **kwargs):
    f1 = argument.element._compiler_dispatch(self, **kwargs)
    f2 = argument.order_by._compiler_dispatch(self, **kwargs)
    assert f1[-1] == ')', f'Must end in paren: {f1}'
    return f'{f1[:-1]} ORDER BY {f2})'
sa.sql.compiler.SQLCompiler.visit_argumentorderby = _visit_argumentorderby
###############################################################################


# Limit to only those which are id_lowest, for safety
_attrs_tv = sa.func.jsonb_each(sch.AttrsBase.attrs).table_valued('key', 'value').lateral()
_attrs_decomposed = (
        sa.select(sch.AttrsBase.id, _attrs_tv.c.key, _attrs_tv.c.value)
        .select_from(sch.AttrsBase)
        .join(_attrs_tv, sa.true())
        ).subquery()
_attrs_sources = (
        sa.select(sch.ComputedAttrs.obj_id.label('ent_id'), sch.ComputedAttrs.id)
        .union_all(sa.select(sch.AttrsBase.id.label('ent_id'), sch.AttrsBase.id))
        ).subquery()
_entity_computed_attrs = (
        sa.select(
            sch.EntityFusion.id_lowest.label('id'),
            # Note use of a sort order here to guarantee stable results. Later
            # batches will always have larger ComputedAttrs.id values due to
            # incrementing PK.
            (
                # Can be null if empty object, so coalesce and strip_nulls outside
                sa.func.jsonb_strip_nulls(
                    _ArgumentOrderBy(
                        sa.func.jsonb_object_agg(
                            sa.func.coalesce(_attrs_decomposed.c.key, '<no attrs found>'),
                            _attrs_decomposed.c.value),
                    _attrs_decomposed.c.id.asc()))
                .label('computed_attrs')),
        )
        .select_from(sch.EntityFusion)
        .join(_attrs_sources, _attrs_sources.c.ent_id == sch.EntityFusion.id_other)
        # SQLalchemy docs seem wrong -- need non-true join condition to suppress
        # warning.
        .join(_attrs_decomposed, _attrs_decomposed.c.id == _attrs_sources.c.id,
            # Can be null -- empty object, for instance
            isouter=True)
        .group_by(sch.EntityFusion.id_lowest)
        ).subquery()
_entity_is_lowest = sa.select(sch.EntityFusion).where(
        sch.EntityFusion.id_lowest == sch.Entity.id).exists()
_entity_query = (
        sa.select(
            # Purposely omit fields like `batch_id` which do not apply to fused
            # entity.
            sch.Entity.id,
            sch.Entity.name,
            sch.Entity.type,
            _entity_computed_attrs.c.computed_attrs.label('attrs'),
        )
        .where(_entity_is_lowest)
        .join(_entity_computed_attrs, _entity_computed_attrs.c.id == sch.Entity.id)
        ).subquery()
@dataclasses.dataclass
class FusedEntity(sch.Base, sch.DataClassMixin):
    __table__ = _entity_query
    __repr__ = sch.Entity.__repr__

    id: int
    name: str
    type: sch.EntityTypeEnum
    attrs: Dict[str, Any]

    fusions = sa.orm.relationship('EntityFusion',
            primaryjoin='EntityFusion.id_lowest == FusedEntity.id',
            viewonly=True)

    @property
    def attrs_sources(self):
        """Returns a Query object (can call .all() on it) which has, in order
        of reverse importance, each object whose `attrs` field is merged into
        this `FusedEntity`'s.
        """
        sess = sa.orm.object_session(self)

        ids = (sa.select(sch.AttrsBase.id.label('ent_id'), sch.AttrsBase.id)
                .union_all(
                    sa.select(sch.ComputedAttrs.obj_id.label('ent_id'),
                        sch.ComputedAttrs.id))).cte()
        subq = (
                sess.query(sa.orm.with_polymorphic(sch.AttrsBase,
                    [sch.Entity, sch.ComputedAttrs]))
                .join(ids, ids.c.id == sch.AttrsBase.id)
                .join(sch.EntityFusion, sch.EntityFusion.id_other == ids.c.ent_id)
                .where(sch.EntityFusion.id_lowest == self.id)
                )
        return subq.order_by(sch.AttrsBase.id.asc())


    def obs_hops(self, k, time_min=None, time_max=None):
        """Returns an array of all observations (which include links to entities)
        within `k` hops of this entity. Optionally, those observations may be
        bounded by time range.
        """

        # Use -1 as it will not exist in the DB
        q = sa.select(sa.literal(-1).label('id'), sa.literal(self.id).label('id_linked'))
        limit = []
        if time_min is not None:
            if isinstance(time_min, float):
                time_min = datetime.datetime.fromtimestamp(time_min)
            limit.append(FusedObservation.time >= time_min)
        if time_max is not None:
            if isinstance(time_max, float):
                time_max = datetime.datetime.fromtimestamp(time_max)
            limit.append(FusedObservation.time < time_max)
        for i in range(k):
            q = sa.union(q,
                    sa.select(FusedObservation.id, FusedObservation.src_id).where(
                        FusedObservation.dst_id == q.c.id_linked, *limit)
                    .union(
                        sa.select(FusedObservation.id, FusedObservation.dst_id).where(
                            FusedObservation.src_id == q.c.id_linked, *limit))
                    )
        sess = sa.orm.object_session(self)
        obj_query = sess.query(FusedObservation).where(
                FusedObservation.id == q.c.id)
        # Pull in the entities efficiently
        obj_query = obj_query.options(sa.orm.selectinload(FusedObservation.dst))
        obj_query = obj_query.options(sa.orm.selectinload(FusedObservation.src))
        return obj_query.all()


# Rewrite FusedObservation s.t. the dst_id and src_id fields are replaced
# correctly with fused versions.
_O = sch.Observation
_F1 = sa.orm.aliased(sch.EntityFusion)
_F2 = sa.orm.aliased(sch.EntityFusion)
_obs_computed_attrs = (
        sa.select(
            _O.id,
            # Note use of a sort order here to guarantee stable results. Later
            # batches will always have larger ComputedAttrs.id values due to
            # incrementing PK.
            (
                # Can be null if empty object, so coalesce and strip_nulls outside
                sa.func.jsonb_strip_nulls(
                    _ArgumentOrderBy(
                        sa.func.jsonb_object_agg(
                            sa.func.coalesce(_attrs_decomposed.c.key, '<no attrs found>'),
                            _attrs_decomposed.c.value),
                    _attrs_decomposed.c.id.asc()))
                .label('computed_attrs')),
        )
        .select_from(_O)
        .join(_attrs_sources, _attrs_sources.c.ent_id == _O.id)
        # SQLalchemy docs seem wrong -- need non-true join condition to suppress
        # warning.
        .join(_attrs_decomposed, _attrs_decomposed.c.id == _attrs_sources.c.id,
            # Can be null -- empty object, for instance
            isouter=True)
        .group_by(_O.id)
        ).subquery()
_obs_query = (
        sa.select(_O.id, _O.batch_id, _O.type, _O.time,
            _obs_computed_attrs.c.computed_attrs.label('attrs'),
            _F1.id_lowest.label('src_id'), _F2.id_lowest.label('dst_id'))
        .select_from(_O)
        .join(_obs_computed_attrs, _obs_computed_attrs.c.id == _O.id)
        .join(_F1, _O.src_id == _F1.id_other)
        .join(_F2, _O.dst_id == _F2.id_other)
        ).subquery()
@dataclasses.dataclass
class FusedObservation(sch.Base, sch.DataClassMixin):
    __table__ = _obs_query
    __repr__ = sch.Observation.__repr__

    id: int
    batch_id: int
    type: sch.ObservationTypeEnum
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

    @property
    def attrs_sources(self):
        """Returns a Query object (can call .all() on it) which has, in order
        of reverse importance, each object whose `attrs` field is merged into
        this `FusedObservation`'s.
        """
        sess = sa.orm.object_session(self)

        ids = (sa.select(sch.AttrsBase.id.label('ent_id'), sch.AttrsBase.id)
                .union_all(
                    sa.select(sch.ComputedAttrs.obj_id.label('ent_id'),
                        sch.ComputedAttrs.id))).cte()
        subq = (
                sess.query(sa.orm.with_polymorphic(sch.AttrsBase,
                    [sch.Observation, sch.ComputedAttrs]))
                .join(ids, ids.c.id == sch.AttrsBase.id)
                .where(ids.c.ent_id == self.id)
                )
        return subq.order_by(sch.AttrsBase.id.asc())

