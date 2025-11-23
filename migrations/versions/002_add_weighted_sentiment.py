"""Add weighted sentiment and comment_likes columns

Revision ID: 002_weighted_sentiment
Revises: 001_initial
Create Date: 2024-11-22

This migration adds support for like-weighted sentiment analysis.
Adds comment_likes (alias for likes) and weighted_sentiment columns.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import String, Float, Integer, DateTime, Text

# revision identifiers
revision = '002_weighted_sentiment'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade():
    """Add weighted sentiment columns"""
    # Add comment_likes column (alias for likes, for clarity)
    try:
        op.add_column('comments', sa.Column('comment_likes', Integer))
        # Copy data from likes column if it exists
        op.execute("UPDATE comments SET comment_likes = likes WHERE comment_likes IS NULL")
    except Exception:
        # Column might already exist, ignore
        pass
    
    # Add weighted_sentiment column
    try:
        op.add_column('comments', sa.Column('weighted_sentiment', Float))
    except Exception:
        # Column might already exist, ignore
        pass
    
    # Add post_caption column if it doesn't exist
    try:
        op.add_column('comments', sa.Column('post_caption', Text))
    except Exception:
        # Column might already exist, ignore
        pass


def downgrade():
    """Remove weighted sentiment columns"""
    try:
        op.drop_column('comments', 'weighted_sentiment')
    except Exception:
        pass
    
    try:
        op.drop_column('comments', 'comment_likes')
    except Exception:
        pass
    
    try:
        op.drop_column('comments', 'post_caption')
    except Exception:
        pass

