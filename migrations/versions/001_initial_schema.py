"""Initial database schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-12-XX

This migration creates the initial comments table schema.
Matches the current CommentIngester._init_database() structure.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import String, Float, Integer, DateTime, Text

# revision identifiers
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial comments table"""
    op.create_table(
        'comments',
        sa.Column('comment_id', String(50), primary_key=True),
        sa.Column('platform', String(20)),
        sa.Column('username', String(200)),
        sa.Column('comment_text', Text),
        sa.Column('timestamp', DateTime),
        sa.Column('likes', Integer),
        sa.Column('post_id', String(50)),
        sa.Column('post_subject', String(200)),
        sa.Column('post_url', String(500)),
        sa.Column('post_caption', Text),
        sa.Column('primary_emotion', String(50)),
        sa.Column('sentiment_score', Float),
        sa.Column('is_sarcastic', Integer),  # SQLite doesn't have boolean
        sa.Column('secondary_emotions', Text),  # JSON string
        sa.Column('mentioned_entities', Text),  # JSON string
    )
    
    # Create indexes for common queries
    op.create_index('idx_comments_timestamp', 'comments', ['timestamp'])
    op.create_index('idx_comments_platform', 'comments', ['platform'])
    op.create_index('idx_comments_post_id', 'comments', ['post_id'])


def downgrade():
    """Drop comments table"""
    op.drop_index('idx_comments_post_id', 'comments')
    op.drop_index('idx_comments_platform', 'comments')
    op.drop_index('idx_comments_timestamp', 'comments')
    op.drop_table('comments')

