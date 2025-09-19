"""auto-merge heads

Revision ID: 5a8a46420f63
Revises: e8e8a60a518f
Create Date: 2025-09-16 10:11:34.993137

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5a8a46420f63'
down_revision: Union[str, Sequence[str], None] = 'e8e8a60a518f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
