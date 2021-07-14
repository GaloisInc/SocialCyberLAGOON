"""LAGOON's core schema.

Note for graph queries: https://stackoverflow.com/a/52489739
"""

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

@dataclasses.dataclass
class Batch(Base):
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


class EntityTypeEnum(enum.Enum):
    '''An enum of entity types. Standardized across all ingestions for
    consistency.
    '''
    git_commit = 'git_commit'
    file = 'file'
    message = 'message'
    person = 'person'


class ObservationTypeEnum(enum.Enum):
    '''An enum of observation types.
    '''
    attached_to = 'attached_to'
    committed = 'committed'
    created = 'created'
    modified = 'modified'


@dataclasses.dataclass
class Entity(Base):
    '''An entity within the extracted information.

    Attributes:
        obs_as_dst: Backref to all `Observation` where this entity is `dst`.
        obs_as_src: Backref to all `Observation` where this entity is `src`.
    '''
    __tablename__ = 'entity'
    def __repr__(self):
        cls = self.__class__
        return f'<{cls.__module__}.{cls.__name__} {self.id}: {self.type} {self.name}>'

    id: int = sa.Column(sa.Integer, primary_key=True)

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

    # Any attributes -- ways of describing this entity's properties (e.g.,
    # e-mail, twitter handle, an email vs a user, etc)
    attrs: Dict[str, Any] = sa.Column(DbJson(), nullable=False, default=lambda: {})

    # Backlinks to observations -- created via `backref` in sqlalchemy
    # obs_as_dst
    # obs_as_src

    # Convenience observations -- earliest and latest, may be either src or dst
    @property
    def obs_earliest(self):
        s = self.obs_as_src.order_by(Observation.time).limit(1).scalar()
        d = self.obs_as_dst.order_by(Observation.time).limit(1).scalar()
        if s and d:
            if s.time < d.time:
                return s
            return d
        elif s:
            return s
        return d
    @property
    def obs_latest(self):
        s = self.obs_as_src.order_by(Observation.time.desc()).limit(1).scalar()
        d = self.obs_as_dst.order_by(Observation.time.desc()).limit(1).scalar()
        if s and d:
            if s.time > d.time:
                return s
            return d
        elif s:
            return s
        return d



@dataclasses.dataclass
class Observation(Base):
    '''An observation that was extracted.'''
    __tablename__ = 'observation'
    def __repr__(self, nodb=False):
        cls = self.__class__
        r = [f'<{cls.__module__}.{cls.__name__} {self.id}: ({self.type}']
        if self.value is not None:
            r.append(f'={self.value}')
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

    id: int = sa.Column(sa.Integer, primary_key=True)

    # For group deletion -- identifier of batch
    batch_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('batch.id', ondelete='CASCADE'),
            nullable=False,
            index=True)
    batch = sa.orm.relationship('Batch', back_populates='observations')

    # Basically "key: value" pair. Importantly, "type" is basically the "class"
    # of the observation, and so any idea of directionality or other qualities
    # of the observation other than its value should be attached to that
    # information.
    type: ObservationTypeEnum = sa.Column(DbEnum(ObservationTypeEnum), nullable=False)
    value: float = sa.Column(sa.Float)

    # UTC time of observation
    time: datetime.datetime = sa.Column(sa.DateTime, nullable=False)

    # Miscellaneous attributes
    attrs: Dict[str, Any] = sa.Column(DbJson(), nullable=False,
            default=lambda: {})

    # Entities participating in this observation
    dst_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id', ondelete='CASCADE'),
            nullable=False)
    dst = sa.orm.relationship('Entity',
            backref=sa.orm.backref('obs_as_dst', lazy='dynamic'),
            primaryjoin='Entity.id==Observation.dst_id',
            foreign_keys=dst_id)
    src_id: int = sa.Column(sa.Integer,
            sa.ForeignKey('entity.id', ondelete='CASCADE'),
            nullable=False)
    src = sa.orm.relationship('Entity',
            backref=sa.orm.backref('obs_as_src', lazy='dynamic'),
            primaryjoin='Entity.id==Observation.src_id',
            foreign_keys=src_id)

    __table_args__ = (
            sa.Index('idx_observation_dst_by_timestamp', 'dst_id', 'time'),
            sa.Index('idx_observation_src_by_timestamp', 'src_id', 'time'),
    )

