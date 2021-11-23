"""LAGOON's core schema.

Note for graph queries: https://stackoverflow.com/a/52489739
"""

import arrow
import dataclasses
import datetime
import enum
import sqlalchemy as sa
from sqlalchemy.orm import relationship, registry
import sqlalchemy_json as sj
from typing import Any, Dict

mapper_registry = registry()
Base = mapper_registry.generate_base()

DbEnum = lambda e: sa.Enum(e, native_enum=False, length=255)
DbJson = lambda: sj.mutable_json_type(dbtype=sa.dialects.postgresql.JSONB,
        nested=True)

class DataClassMixin:
    def asdict(self):
        r = dataclasses.asdict(self)
        def encobjs(r):
            # Recursively remove sqlalchemy wrapper magic
            if isinstance(r, dict):
                return {k: encobjs(v) for k, v in r.items()}
            elif isinstance(r, list):
                return [encobjs(v) for v in r]
            return r
        return encobjs(r)


@dataclasses.dataclass
class Batch(Base, DataClassMixin):
    '''A batch of imported data. Ingesting could be expensive, and should be
    cached where possible. This table is used to track metadata for purging old
    information.

    Note that resource ingestion is considered a linear process -- we don't
    support forked batches at the moment. That's why the PK is
    resource + ingest_time. Similarly, though the responsibility is on the user,
    the `revision` field should be monotonic.
    '''
    __tablename__ = 'batch'

    id: int = sa.Column(sa.Integer, primary_key=True)

    # Resource combines ingest type AND specific resource being imported.
    resource: str = sa.Column(sa.String, nullable=False)
    ingest_time: datetime.datetime = sa.Column(sa.DateTime,
            default=datetime.datetime.utcnow)

    # Revision is for documenting "how much" data was ingested; useful for
    # not re-polling information which has already been fetched. Probably should
    # be monotonic for some definition.
    revision: str = sa.Column(sa.String)

    computed_attrs = sa.orm.relationship('ComputedAttrs', back_populates='batch',
            lazy='dynamic', cascade='all, delete')
    entities = sa.orm.relationship('Entity', back_populates='batch',
            lazy='dynamic', cascade='all, delete')
    observations = sa.orm.relationship('Observation', back_populates='batch',
            lazy='dynamic', cascade='all, delete')

    __table_args__ = (
            sa.Index('idx_batch_pkey', 'resource', 'ingest_time', unique=True),
    )

    @classmethod
    def cls_reset_resource(cls, resource, session):
        """Delete all entities / objects / information relating to batches on
        the specified resource.
        """
        # SQL cascades cause other objects to be deleted.
        session.execute(sa.delete(cls).where(cls.resource == resource))


class SuperTypeEnum(enum.Enum):
    computed_attrs = 'computed_attrs'
    entity = 'entity'
    observation = 'observation'


class EntityTypeEnum(enum.Enum):
    '''An enum of entity types. Standardized across all ingestions for
    consistency.
    '''
    git_commit = 'git_commit'
    file = 'file'
    message = 'message'
    person = 'person'
    pep = 'pep'


class ObservationTypeEnum(enum.Enum):
    '''An enum of observation types.
    '''
    attached_to = 'attached_to'
    committed = 'committed'
    created = 'created'
    message_cc = 'message_cc'
    message_from = 'message_from'
    message_ref = 'message_ref'
    message_to = 'message_to'
    modified = 'modified'
    superseded_by = 'superseded_by'
    requires = 'requires'


@dataclasses.dataclass
class AttrsBase(Base, DataClassMixin):
    '''Basically, because both Entities and Observations store their
    type-specific data in an `attrs` field, and it can sometimes be useful to
    retract / amend this information in subsequent ingests, we use `AttrsBase`
    as a parent type for each.

    f
    '''
    __tablename__ = 'attrs_base'

    id: int = sa.Column(sa.Integer, primary_key=True)

    super_type: str = sa.Column(DbEnum(SuperTypeEnum), nullable=False)

    # Miscellaneous attributes describing this type (e.g., for a user, e-mail,
    # twitter handle, etc; for an observation, the number of lines removed from
    # a file by a git commit, etc)
    attrs: Dict[str, Any] = sa.Column(DbJson(), nullable=False,
            default=lambda: {})

    __mapper_args__ = {
            'polymorphic_on': 'super_type',
    }



@dataclasses.dataclass
class Entity(AttrsBase, DataClassMixin):
    '''An entity within the extracted information.

    For general data processing, to get a neighborhood around a specific entity,
    Use :class:`lagoon.db.schema_fused.FusedEntity` instead, and do something
    like:

    ```python
    import lagoon.db.connection.get_session as get_session
    import lagoon.db.schema as sch
    import arrow
    import sqlalchemy as sa
    start = arrow.get('20200401').datetime
    end = arrow.get('20200801').datetime
    with get_session() as sess:
        obj = sess.execute(sa.select(sch.FusedEntity).limit(1)).scalar()
        # Get all FusedObservation objects, and their corresponding entities,
        # within 2 edge hops of this object
        obs = obj.obs_hops(2, time_min=start, time_max=end)

        # Can look at entities attached to each edge as:
        obs[0].dst, obs[0].src
    ```

    Attributes:
        obs_as_dst: Backref to all `Observation` where this entity is `dst`.
        obs_as_src: Backref to all `Observation` where this entity is `src`.
    '''
    __tablename__ = 'entity'
    def __repr__(self):
        cls = self.__class__
        return f'<{cls.__module__}.{cls.__name__} {self.id}: {self.type} {self.name}>'

    id: int = sa.Column(sa.Integer, sa.ForeignKey('attrs_base.id'),
            primary_key=True)

    # The batch which created this entity
    batch_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('batch.id', ondelete='CASCADE'),
            nullable=False,
            index=True)
    batch = sa.orm.relationship('Batch', back_populates='entities')

    # Display name
    name: str = sa.Column(sa.String, nullable=False)

    # Unique, discretized type of entity. This exists to force ingest processes
    # to coalesce their definitions of
    type: EntityTypeEnum = sa.Column(DbEnum(EntityTypeEnum), nullable=False)

    # Backlinks to observations -- created via `backref` in sqlalchemy
    # obs_as_dst
    # obs_as_src

    __mapper_args__ = {
            'polymorphic_identity': SuperTypeEnum.entity,
    }

    def fused(self):
        """Returns a :class:`FusedEntity` which corresponds to this entity,
        but fused.
        """
        fe = sa.orm.aliased(FusedEntity)

        return sa.orm.object_session(self).execute(
                sa.select(fe)
                .select_from(EntityFusion)
                .where(EntityFusion.id_other == self.id)
                .join(fe, fe.id == EntityFusion.id_lowest)
                ).scalar()



@dataclasses.dataclass
class Observation(AttrsBase, DataClassMixin):
    '''An observation that was extracted.'''
    __tablename__ = 'observation'
    def __repr__(self, nodb=False):
        cls = self.__class__
        r = [f'<{cls.__module__}.{cls.__name__} {self.id}: ({self.type}']
        time_str = arrow.get(self.time).format('YYYY-MM-DD')
        r.append(f'@{time_str}')
        if not nodb:
            s = self.src_id
            # Prevent DB hits in repr
            if 'src' in self.__dict__:
                s = self.src
            d = self.dst_id
            if 'dst' in self.__dict__:
                d = self.dst
            r.append(f', {s}, {d})>')
        return ''.join(r)

    id: int = sa.Column(sa.Integer, sa.ForeignKey('attrs_base.id'),
            primary_key=True)

    # For group deletion -- identifier of batch
    batch_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('batch.id', ondelete='CASCADE'),
            nullable=False,
            index=True)
    batch = sa.orm.relationship('Batch', back_populates='observations')

    # Basically an observation specifies some typed relationship.
    # Importantly, "type" is basically the "class" of the observation, and so
    # any idea of directionality or other qualities of the observation
    # that might be present should be attached to that information.
    type: ObservationTypeEnum = sa.Column(DbEnum(ObservationTypeEnum), nullable=False)

    # UTC time of observation
    time: datetime.datetime = sa.Column(sa.DateTime, nullable=False)

    # Entities participating in this observation
    dst_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id'),
            nullable=False)
    dst = sa.orm.relationship('Entity',
            backref=sa.orm.backref('obs_as_dst', lazy='dynamic'),
            primaryjoin='Entity.id==Observation.dst_id',
            foreign_keys=dst_id)
    src_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id'),
            nullable=False)
    src = sa.orm.relationship('Entity',
            backref=sa.orm.backref('obs_as_src', lazy='dynamic'),
            primaryjoin='Entity.id==Observation.src_id',
            foreign_keys=src_id)

    __table_args__ = (
            sa.Index('idx_observation_dst_by_timestamp', 'dst_id', 'time'),
            sa.Index('idx_observation_src_by_timestamp', 'src_id', 'time'),
    )
    __mapper_args__ = {
            'polymorphic_identity': SuperTypeEnum.observation,
    }


@dataclasses.dataclass
class ComputedAttrs(AttrsBase, DataClassMixin):
    '''A record which modified entities or observations from a previously
    ingested batch.
    '''
    __tablename__ = 'computed_attrs'

    id: int = sa.Column(sa.Integer, sa.ForeignKey('attrs_base.id'),
            primary_key=True)

    # The batch which created this ComputedAttrs
    batch_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('batch.id', ondelete='CASCADE'),
            nullable=False,
            index=True)
    batch = sa.orm.relationship('Batch', back_populates='computed_attrs')

    obj_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('attrs_base.id'),
            nullable=False,
            index=True)
    obj = sa.orm.relationship('AttrsBase',
            backref=sa.orm.backref('computed_attrs', lazy='dynamic',
                order_by='ComputedAttrs.id'),
            primaryjoin='AttrsBase.id==ComputedAttrs.obj_id',
            foreign_keys=[obj_id])

    __table_args__ = (
            #sa.Index('idx_computed_attrs_attrs', ...)
    )
    __mapper_args__ = {
            'polymorphic_identity': SuperTypeEnum.computed_attrs,
            'inherit_condition': id == AttrsBase.id,
    }



@dataclasses.dataclass
class EntityFusion(Base, DataClassMixin):
    '''A record of entities being fused.

    Note that this table is PK'd by id_other -- that's because we want each
    entity to belong to 0 or 1 groups.
    '''
    __tablename__ = 'entity_fusion'

    id_lowest: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id'),
            index=True)
    lowest = sa.orm.relationship('Entity',
            foreign_keys=[id_lowest])

    id_other: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id'),
            autoincrement=False,
            primary_key=True)
    other = sa.orm.relationship('Entity',
            foreign_keys=[id_other])

    comment: str = sa.Column(sa.String)



"""
@dataclasses.dataclass
class EntityFusionManual(Base):
    '''A record of known entity relationships which guide / force those
    relationships during the fusion process.

    NOT CURRENTLY IMPLEMENTED. So commented out.
    '''
    __tablename__ = 'entity_fusion_manual'
    id: int = sa.Column(sa.Integer, primary_key=True)

    id_a: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id'),
            nullable=False,
            index=True)
    id_b: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id'),
            nullable=False,
            index=True)
    same: bool = sa.Column(sa.Boolean)
    comment: str = sa.Column(sa.String)
    comment_time: datetime.datetime = sa.Column(sa.DateTime,
            default=datetime.datetime.utcnow)
"""


from .schema_fused import FusedEntity, FusedObservation

