"""Rename password to hashed_password on users table

Revision ID: 994c4c83dc8a
Revises: 2ad5d006620f
Create Date: 2026-08-28 15:00:09.786341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '994c4c83dc8a'
down_revision: Union[str, None] = '2ad5d006620f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('users', 'password', new_column_name='hashed_password')


def downgrade() -> None:
    op.alter_column('users', 'hashed_password', new_column_name='password')
