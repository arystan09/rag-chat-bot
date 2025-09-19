"""auto-merge heads

Revision ID: e8e8a60a518f
Revises: add_answer_to_query_logs, add_url_field
Create Date: 2025-09-16 10:11:31.512736

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8e8a60a518f'
down_revision: Union[str, Sequence[str], None] = ('add_answer_to_query_logs', 'add_url_field')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
