"""empty message

Revision ID: 9149c9ca3367
Revises: 
Create Date: 2021-07-09 07:56:45.421774

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '9149c9ca3367'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('batch',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('resource', sa.String(), nullable=False),
    sa.Column('ingest_time', sa.DateTime(), nullable=True),
    sa.Column('revision', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_batch_pkey', 'batch', ['resource', 'ingest_time'], unique=True)
    op.create_table('entity',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('batch_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('type', sa.Enum('git_commit', 'file', 'message', 'person', name='entitytypeenum', native_enum=False, length=255), nullable=False),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.ForeignKeyConstraint(['batch_id'], ['batch.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_entity_batch_id'), 'entity', ['batch_id'], unique=False)
    op.create_table('observation',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('batch_id', sa.Integer(), nullable=False),
    sa.Column('type', sa.Enum('author', 'modified', name='observationtypeenum', native_enum=False, length=255), nullable=False),
    sa.Column('value', sa.Float(), nullable=True),
    sa.Column('time', sa.DateTime(), nullable=False),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('dst_id', sa.Integer(), nullable=False),
    sa.Column('src_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['batch_id'], ['batch.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['dst_id'], ['entity.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['src_id'], ['entity.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_observation_dst_by_timestamp', 'observation', ['dst_id', 'time', 'id'], unique=False)
    op.create_index('idx_observation_src_by_timestamp', 'observation', ['src_id', 'time', 'id'], unique=False)
    op.create_index(op.f('ix_observation_batch_id'), 'observation', ['batch_id'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_observation_batch_id'), table_name='observation')
    op.drop_index('idx_observation_src_by_timestamp', table_name='observation')
    op.drop_index('idx_observation_dst_by_timestamp', table_name='observation')
    op.drop_table('observation')
    op.drop_index(op.f('ix_entity_batch_id'), table_name='entity')
    op.drop_table('entity')
    op.drop_index('idx_batch_pkey', table_name='batch')
    op.drop_table('batch')
    # ### end Alembic commands ###
