# Database Migration Guide

This guide explains how to manage database schema changes using Alembic.

## Overview

As the database schema evolves (adding columns, indexes, etc.), migrations ensure existing databases are updated safely without data loss.

**Critical:** Always test migrations on a backup first!

## Running Migrations

### Initial Setup

If this is the first time setting up migrations:

```bash
# The initial migration (001_initial_schema.py) matches the current schema
alembic upgrade head
```

This creates the `comments` table if it doesn't exist.

### Creating New Migrations

When you need to change the database schema:

1. **Modify the schema** in code (e.g., `CommentIngester._init_database()`)

2. **Create a new migration:**

```bash
alembic revision -m "add_new_column_description"
```

3. **Edit the generated migration file** in `migrations/versions/`:

```python
def upgrade():
    op.add_column('comments', sa.Column('new_column', sa.String(200)))
    op.create_index('idx_comments_new_column', 'comments', ['new_column'])

def downgrade():
    op.drop_index('idx_comments_new_column', 'comments')
    op.drop_column('comments', 'new_column')
```

4. **Test the migration:**

```bash
# Upgrade
alembic upgrade head

# Downgrade (if needed)
alembic downgrade -1
```

### Applying Migrations

In production:

```bash
# Check current version
alembic current

# Upgrade to latest
alembic upgrade head

# Upgrade one step
alembic upgrade +1

# Downgrade one step (use with caution!)
alembic downgrade -1
```

## Migration Best Practices

### 1. **Always Create Both upgrade() and downgrade()**

```python
def upgrade():
    # Add new column
    op.add_column('comments', sa.Column('new_field', sa.String(100)))

def downgrade():
    # Remove column (reverses upgrade)
    op.drop_column('comments', 'new_field')
```

### 2. **Handle Data Migrations Carefully**

If adding a NOT NULL column to existing data:

```python
def upgrade():
    # Add column as nullable first
    op.add_column('comments', sa.Column('new_field', sa.String(100), nullable=True))
    
    # Populate with default values
    op.execute("UPDATE comments SET new_field = 'default' WHERE new_field IS NULL")
    
    # Make NOT NULL (if needed)
    op.alter_column('comments', 'new_field', nullable=False)
```

### 3. **Create Indexes for Performance**

```python
def upgrade():
    op.create_index('idx_comments_timestamp', 'comments', ['timestamp'])
```

### 4. **Test Migrations on Backup First**

```bash
# Backup database
cp et_intel/data/database/et_intelligence.db et_intel/data/database/et_intelligence.db.backup

# Run migration on backup
alembic upgrade head

# Verify data integrity
python -m et_intel.cli.cli --stats
```

## Common Migration Scenarios

### Adding a New Column

```python
def upgrade():
    op.add_column('comments', 
        sa.Column('engagement_score', sa.Float(), nullable=True)
    )

def downgrade():
    op.drop_column('comments', 'engagement_score')
```

### Adding an Index

```python
def upgrade():
    op.create_index('idx_comments_sentiment', 'comments', ['sentiment_score'])

def downgrade():
    op.drop_index('idx_comments_sentiment', 'comments')
```

### Modifying a Column Type

```python
def upgrade():
    # SQLite requires creating new table and copying data
    op.execute("""
        ALTER TABLE comments 
        RENAME TO comments_old;
        
        CREATE TABLE comments (
            comment_id VARCHAR(50) PRIMARY KEY,
            -- ... other columns ...
            likes INTEGER
        );
        
        INSERT INTO comments SELECT * FROM comments_old;
        
        DROP TABLE comments_old;
    """)

def downgrade():
    # Reverse the change
    pass
```

### Adding a New Table

```python
def upgrade():
    op.create_table(
        'entity_tracking',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('entity_name', sa.String(200)),
        sa.Column('mention_count', sa.Integer()),
        sa.Column('last_updated', sa.DateTime())
    )

def downgrade():
    op.drop_table('entity_tracking')
```

## Troubleshooting

### Migration fails with "table already exists"

The table was created outside of Alembic. Options:

1. **Mark as already applied** (if schema matches):
```bash
alembic stamp head
```

2. **Create migration that matches existing schema**:
```bash
alembic revision --autogenerate -m "match_existing_schema"
```

### Migration fails with "can't downgrade"

Check if `downgrade()` function is properly implemented. If data will be lost, consider making it a no-op:

```python
def downgrade():
    # Data loss would occur, so we can't safely downgrade
    pass  # or raise NotImplementedError("Cannot downgrade safely")
```

### Database is locked

SQLite locks the database during migrations. Ensure:
- No other processes are accessing the database
- Close any open connections
- Wait for long-running queries to finish

## Migration History

View all migrations:

```bash
alembic history
```

View specific migration:

```bash
alembic show <revision_id>
```

## Production Checklist

Before running migrations in production:

- [ ] Test migration on backup database
- [ ] Backup production database
- [ ] Verify migration file is correct
- [ ] Schedule downtime if needed (for large migrations)
- [ ] Monitor logs during migration
- [ ] Verify data integrity after migration
- [ ] Test application functionality

## Integration with Docker

When running in Docker:

```bash
# Run migrations in container
docker-compose run --rm et-intel alembic upgrade head

# Or add to entrypoint script
docker-compose run --rm et-intel sh -c "alembic upgrade head && python -m et_intel.cli.cli --batch"
```

