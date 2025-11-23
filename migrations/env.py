"""
Alembic migration environment configuration
Handles database connection and migration context
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import et_intel.config as app_config

# This is the Alembic Config object
config_dict = context.config

# Interpret the config file for Python logging
if config_dict.config_file_name:
    fileConfig(config_dict.config_file_name)

# Set SQLite database path
database_url = f"sqlite:///{app_config.DB_DIR / 'et_intelligence.db'}"
config_dict.set_main_option("sqlalchemy.url", database_url)

# Target metadata (for auto-generate migrations)
target_metadata = None  # We'll manually create migrations

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config_dict.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config_dict.get_section(config_dict.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

