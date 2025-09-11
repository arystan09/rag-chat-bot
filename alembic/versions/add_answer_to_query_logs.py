"""Add answer column to query_logs

Revision ID: add_answer_to_query_logs
Revises: d85a577aa4c9
Create Date: 2025-09-10 20:35:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_answer_to_query_logs'
down_revision = 'd85a577aa4c9'
branch_labels = None
depends_on = None


def upgrade():
    """Add answer column to query_logs table."""
    # Check if column already exists
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    columns = [col['name'] for col in inspector.get_columns('query_logs')]
    
    if 'answer' not in columns:
        op.add_column('query_logs', sa.Column('answer', sa.Text(), nullable=True))


def downgrade():
    """Remove answer column from query_logs table."""
    op.drop_column('query_logs', 'answer')

