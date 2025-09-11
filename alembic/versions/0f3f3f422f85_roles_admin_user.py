"""roles_admin_user

Revision ID: 0f3f3f422f85
Revises: 25408aa19382
Create Date: 2025-09-08 22:30:42.975348

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0f3f3f422f85'
down_revision: Union[str, Sequence[str], None] = 'd85a577aa4c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - change role enum to string with admin/user values."""
    
    # Step 1: Alter column type from enum to varchar
    op.execute("ALTER TABLE users ALTER COLUMN role TYPE VARCHAR(32) USING role::text")
    
    # Step 2: Update existing role values
    op.execute("UPDATE users SET role='admin' WHERE role='client'")
    op.execute("UPDATE users SET role='user' WHERE role='end_user'")
    
    # Step 3: Add check constraint
    op.execute("ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('admin','user'))")
    
    # Step 4: Ensure telegram_id index exists (should already exist)
    op.execute("CREATE INDEX IF NOT EXISTS ix_users_telegram_id ON users (telegram_id)")


def downgrade() -> None:
    """Downgrade schema - remove check constraint."""
    
    # Remove check constraint
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check")
    
    # Note: We don't attempt to convert back to enum to avoid complexity
    # The column remains VARCHAR(32) after downgrade
