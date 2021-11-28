"""empty message

Revision ID: 7d51ac305f89
Revises: da7682ae7b17
Create Date: 2021-11-29 07:44:29.604144

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '7d51ac305f89'
down_revision = 'da7682ae7b17'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('cache_fused_entity',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('type', sa.Enum('git_commit', 'file', 'message', 'person', 'pep', name='entitytypeenum', native_enum=False), nullable=False),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('cache_fused_observation',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('batch_id', sa.Integer(), nullable=False),
    sa.Column('type', sa.Enum('attached_to', 'committed', 'created', 'message_cc', 'message_from', 'message_ref', 'message_to', 'modified', 'superseded_by', 'requires', name='observationtypeenum', native_enum=False), nullable=False),
    sa.Column('time', sa.DateTime(), nullable=False),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('dst_id', sa.Integer(), nullable=False),
    sa.Column('src_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['dst_id'], ['cache_fused_entity.id'], ),
    sa.ForeignKeyConstraint(['src_id'], ['cache_fused_entity.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_fused_observation_dst_by_timestamp', 'cache_fused_observation', ['dst_id', 'time'], unique=False)
    op.create_index('idx_fused_observation_src_by_timestamp', 'cache_fused_observation', ['src_id', 'time'], unique=False)
    op.create_index(op.f('ix_cache_fused_observation_batch_id'), 'cache_fused_observation', ['batch_id'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_cache_fused_observation_batch_id'), table_name='cache_fused_observation')
    op.drop_index('idx_fused_observation_src_by_timestamp', table_name='cache_fused_observation')
    op.drop_index('idx_fused_observation_dst_by_timestamp', table_name='cache_fused_observation')
    op.drop_table('cache_fused_observation')
    op.drop_table('cache_fused_entity')
    # ### end Alembic commands ###
