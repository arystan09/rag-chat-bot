"""Add url field to documents table

Revision ID: add_url_field
Revises: 0f3f3f422f85
Create Date: 2025-09-10 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_url_field'
down_revision: Union[str, Sequence[str], None] = '0f3f3f422f85'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add url field to documents table."""
    op.add_column('documents', sa.Column('url', sa.String(500), nullable=True))


def downgrade() -> None:
    """Remove url field from documents table."""
    op.drop_column('documents', 'url')




