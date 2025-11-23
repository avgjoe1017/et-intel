"""
CSV Ingestion Module
Handles importing comments from Instagram/YouTube CSV exports
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, List, Optional
from .. import config

class CommentIngester:
    """
    Ingests comment data from CSV files and standardizes format
    """
    
    def __init__(self):
        self.uploads_dir = config.UPLOADS_DIR
        self.processed_dir = config.PROCESSED_DIR
        
    def ingest_instagram_csv(self, csv_path: str, post_metadata: Dict = None) -> pd.DataFrame:
        """
        Import Instagram comments from CSV
        
        Expected CSV columns (flexible, will auto-detect):
        - username / author / user
        - comment / text / content
        - timestamp / date / created_at
        - likes / like_count (optional)
        
        Args:
            csv_path: Path to CSV file
            post_metadata: Dict with 'post_url', 'post_caption', 'subject' etc.
        
        Returns:
            Standardized DataFrame
        """
        df = pd.read_csv(csv_path)
        
        # Auto-detect column names (case-insensitive)
        col_map = self._detect_columns(df.columns)
        
        # Standardize columns
        standardized = pd.DataFrame({
            'platform': 'instagram',
            'username': df[col_map['username']],
            'comment_text': df[col_map['comment']],
            'timestamp': pd.to_datetime(df[col_map['timestamp']], errors='coerce'),
            'likes': df[col_map.get('likes', pd.Series([0]*len(df)))],
            'post_id': self._generate_post_id(csv_path, post_metadata),
            'post_subject': post_metadata.get('subject', '') if post_metadata else '',
            'post_url': post_metadata.get('post_url', '') if post_metadata else '',
        })
        
        # Add unique comment ID
        standardized['comment_id'] = standardized.apply(
            lambda row: self._generate_comment_id(row), axis=1
        )
        
        # Filter out very short comments
        standardized = standardized[
            standardized['comment_text'].str.len() >= config.MIN_COMMENT_LENGTH
        ]
        
        return standardized
    
    def ingest_youtube_csv(self, csv_path: str, video_metadata: Dict = None) -> pd.DataFrame:
        """
        Import YouTube comments from CSV
        
        Expected columns:
        - Author / Channel Name
        - Comment / Text
        - Published At / Date
        - Likes (optional)
        """
        df = pd.read_csv(csv_path)
        col_map = self._detect_columns(df.columns)
        
        standardized = pd.DataFrame({
            'platform': 'youtube',
            'username': df[col_map['username']],
            'comment_text': df[col_map['comment']],
            'timestamp': pd.to_datetime(df[col_map['timestamp']], errors='coerce'),
            'likes': df[col_map.get('likes', pd.Series([0]*len(df)))],
            'post_id': self._generate_post_id(csv_path, video_metadata),
            'post_subject': video_metadata.get('subject', '') if video_metadata else '',
            'post_url': video_metadata.get('video_url', '') if video_metadata else '',
        })
        
        standardized['comment_id'] = standardized.apply(
            lambda row: self._generate_comment_id(row), axis=1
        )
        
        standardized = standardized[
            standardized['comment_text'].str.len() >= config.MIN_COMMENT_LENGTH
        ]
        
        return standardized
    
    def _detect_columns(self, columns: List[str]) -> Dict[str, str]:
        """Auto-detect column names from CSV headers"""
        columns_lower = [c.lower() for c in columns]
        
        mapping = {}
        
        # Username detection
        for col, col_lower in zip(columns, columns_lower):
            if any(term in col_lower for term in ['username', 'author', 'user', 'channel']):
                mapping['username'] = col
                break
        
        # Comment text detection
        for col, col_lower in zip(columns, columns_lower):
            if any(term in col_lower for term in ['comment', 'text', 'content', 'message']):
                mapping['comment'] = col
                break
        
        # Timestamp detection
        for col, col_lower in zip(columns, columns_lower):
            if any(term in col_lower for term in ['timestamp', 'date', 'created', 'published', 'time']):
                mapping['timestamp'] = col
                break
        
        # Likes detection (optional)
        for col, col_lower in zip(columns, columns_lower):
            if any(term in col_lower for term in ['like', 'likes']):
                mapping['likes'] = col
                break
        
        # Validate required fields
        required = ['username', 'comment', 'timestamp']
        missing = [r for r in required if r not in mapping]
        if missing:
            raise ValueError(f"Could not detect required columns: {missing}. Available: {columns}")
        
        return mapping
    
    def _generate_post_id(self, csv_path: str, metadata: Optional[Dict]) -> str:
        """Generate unique post ID"""
        if metadata and 'post_url' in metadata:
            return hashlib.md5(metadata['post_url'].encode()).hexdigest()[:12]
        return hashlib.md5(str(csv_path).encode()).hexdigest()[:12]
    
    def _generate_comment_id(self, row: pd.Series) -> str:
        """Generate unique comment ID"""
        unique_str = f"{row['username']}_{row['comment_text'][:50]}_{row['timestamp']}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def save_processed(self, df: pd.DataFrame, filename: str):
        """Save processed comments to database"""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} comments to {output_path}")
        return output_path
    
    def load_all_processed(self) -> pd.DataFrame:
        """Load all processed comment files into single DataFrame"""
        all_files = list(self.processed_dir.glob("*.csv"))
        
        if not all_files:
            return pd.DataFrame()
        
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['comment_id'])
        return pd.DataFrame()



