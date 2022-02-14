"""This file responsible for (re)generating `FusedEntity` and `FusedObservation`.
"""

from lagoon.db.connection import get_session
import lagoon.db.schema as sch
import lagoon.db.schema_fused as schf

import sqlalchemy as sa
import tqdm

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
_entity_sources = schf._attrs_sources_lateral_fn(sch.EntityFusion.id_other)
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
        .join(_entity_sources, sa.true())
        # SQLalchemy docs seem wrong -- need non-true join condition to suppress
        # warning.
        .join(_attrs_decomposed, _attrs_decomposed.c.id == _entity_sources.c.id,
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


# Rewrite FusedObservation s.t. the dst_id and src_id fields are replaced
# correctly with fused versions.
_O = sch.Observation
_O2 = sa.orm.aliased(sch.Observation)
_F1 = sa.orm.aliased(sch.EntityFusion)
_F2 = sa.orm.aliased(sch.EntityFusion)
_obs_sources = schf._attrs_sources_lateral_fn(_O2.id)
_obs_computed_attrs = (
        sa.select(
            _O2.id,
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
        .select_from(_O2)
        .join(_obs_sources, sa.true())
        # SQLalchemy docs seem wrong -- need non-true join condition to suppress
        # warning.
        .join(_attrs_decomposed, _attrs_decomposed.c.id == _obs_sources.c.id,
            # Can be null -- empty object, for instance
            isouter=True)
        .group_by(_O2.id)
        ).subquery()#.lateral()  # lateral() for _O
_obs_query = (
        sa.select(_O.id, _O.batch_id, _O.type, _O.time,
            _obs_computed_attrs.c.computed_attrs.label('attrs'),
            _F1.id_lowest.label('src_id'), _F2.id_lowest.label('dst_id'))
        .select_from(_O)
        .join(_F1, _O.src_id == _F1.id_other)
        .join(_F2, _O.dst_id == _F2.id_other)
        .join(_obs_computed_attrs,
            #sa.true())
            _obs_computed_attrs.c.id == _O.id)
        ).subquery()


def recache():
    """Clears and regenerates `FusedEntity` and `FusedObservation`
    """
    print(f'Re-caching `FusedEntity` and `FusedObservation`')
    with get_session() as sess:
        sess.query(sch.FusedObservation).delete()
        sess.query(sch.FusedEntity).delete()

        print(f'...FusedEntity...')
        sess.execute(sa.insert(sch.FusedEntity).from_select(
                ['id', 'name', 'type', 'attrs'],
                _entity_query))
        print(f'...FusedObservation...')
        sess.execute(sa.insert(sch.FusedObservation).from_select(
                ['id', 'batch_id', 'type', 'time', 'attrs', 'src_id', 'dst_id'],
                _obs_query))

        sess.flush()

        # Assign names
        print(f'...Caching all names for each entity...')
        (sess.query(sch.FusedEntity)
                .update({sch.FusedEntity.cached_names:
                    sa.select(sa.func.string_agg(
                        sch.Entity.name.distinct(),
                        '\n'))
                    .select_from(sch.EntityFusion)
                    .where(sch.EntityFusion.id_lowest == sch.FusedEntity.id)
                    .join(sch.Entity, sch.Entity.id == sch.EntityFusion.id_other)
                    .scalar_subquery()
                }))

