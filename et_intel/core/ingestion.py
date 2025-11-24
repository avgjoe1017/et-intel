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
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import sessionmaker
import sys
from .. import config
from .logging_config import get_logger

logger = get_logger(__name__)

# Import preprocessors
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from preprocess_esuit import detect_esuit_format, preprocess_esuit_csv
    ESUIT_AVAILABLE = True
except ImportError:
    ESUIT_AVAILABLE = False

try:
    from preprocess_apify import detect_apify_format, preprocess_apify_instagram
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False

class CommentIngester:
    """
    Ingests comment data from CSV files and standardizes format
    """
    
    def __init__(self, use_database: bool = True):
        """
        Args:
            use_database: If True, uses SQLite database. If False, uses CSV files only
        """
        self.uploads_dir = config.UPLOADS_DIR
        self.processed_dir = config.PROCESSED_DIR
        self.processed_tracker_path = config.DB_DIR / "processed_csvs.json"
        self._processed_csvs = self._load_processed_tracker()
        
        # SQLite database
        self.use_database = use_database
        self.db_path = config.DB_DIR / "et_intelligence.db"
        self.engine = None
        if self.use_database:
            self._init_database()
        
    def ingest_instagram_csv(self, csv_path: str, post_metadata: Dict = None) -> pd.DataFrame:
        """
        Import Instagram comments from CSV
        Auto-detects and preprocesses ESUIT and Apify format files
        
        Expected CSV columns (flexible, will auto-detect):
        - username / author / user
        - comment / text / content
        - timestamp / date / created_at
        - likes / like_count / comment_like_count (optional)
        
        Args:
            csv_path: Path to CSV file
            post_metadata: Dict with 'post_url', 'post_caption', 'subject' etc.
        
        Returns:
            Standardized DataFrame
        """
        # Check if Apify format and auto-preprocess (check Apify first as it's more specific)
        csv_file = Path(csv_path)
        actual_csv_path = csv_path
        
        if APIFY_AVAILABLE and detect_apify_format(csv_path):
            try:
                logger.info(f"Auto-preprocessing Apify file: {csv_file.name}")
                cleaned_path = preprocess_apify_instagram(csv_path)
                actual_csv_path = cleaned_path
                
                # Update metadata from preprocessed file
                if post_metadata is None:
                    post_metadata = {}
                df_temp = pd.read_csv(cleaned_path)
                if 'post_url' in df_temp.columns and not post_metadata.get('post_url'):
                    post_metadata['post_url'] = df_temp['post_url'].iloc[0] if len(df_temp) > 0 else ''
                if 'post_caption' in df_temp.columns and not post_metadata.get('post_caption'):
                    post_metadata['post_caption'] = df_temp['post_caption'].iloc[0] if len(df_temp) > 0 else ''
            except Exception as e:
                logger.warning(f"Apify preprocessing failed: {e}. Using original file.")
        elif ESUIT_AVAILABLE and detect_esuit_format(csv_path):
            try:
                logger.info(f"Auto-preprocessing ESUIT file: {csv_file.name}")
                cleaned_path = preprocess_esuit_csv(csv_path)
                actual_csv_path = cleaned_path
                
                # Update metadata from preprocessed file
                if post_metadata is None:
                    post_metadata = {}
                df_temp = pd.read_csv(cleaned_path)
                if 'post_url' in df_temp.columns and not post_metadata.get('post_url'):
                    post_metadata['post_url'] = df_temp['post_url'].iloc[0] if len(df_temp) > 0 else ''
                if 'post_caption' in df_temp.columns and not post_metadata.get('post_caption'):
                    post_metadata['post_caption'] = df_temp['post_caption'].iloc[0] if len(df_temp) > 0 else ''
            except Exception as e:
                logger.warning(f"ESUIT preprocessing failed: {e}. Using original file.")
        
        df = pd.read_csv(actual_csv_path)
        
        # Auto-detect column names (case-insensitive)
        col_map = self._detect_columns(df.columns)
        
        # Get likes column - handle both 'likes' and 'comment_like_count' formats
        likes_col = col_map.get('likes')
        if likes_col is None:
            # Try alternative names
            for col in df.columns:
                if 'like' in col.lower() and 'count' in col.lower():
                    likes_col = col
                    break
            if likes_col is None:
                likes_col = pd.Series([0]*len(df))
        
        # Get post_caption from DataFrame if available, otherwise from metadata
        post_caption = ''
        if 'post_caption' in df.columns:
            post_caption = df['post_caption'].iloc[0] if len(df) > 0 else ''
        elif post_metadata and post_metadata.get('post_caption'):
            post_caption = post_metadata['post_caption']
        
        # Get post_url from DataFrame if available, otherwise from metadata
        post_url = ''
        if 'post_url' in df.columns:
            post_url = df['post_url'].iloc[0] if len(df) > 0 else ''
        elif post_metadata and post_metadata.get('post_url'):
            post_url = post_metadata['post_url']

        # Parse timestamps flexibly
        timestamps = pd.to_datetime(df[col_map['timestamp']], errors='coerce')
        
        # If parsing resulted in 1970 dates (often due to Unix timestamp being read as seconds since epoch when it's already a date string, or vice versa), try to fix
        # Check if we have many 1970 dates
        year_counts = timestamps.dt.year.value_counts()
        if 1970 in year_counts and year_counts[1970] > len(df) * 0.5:
            logger.warning("Detected 1970 timestamps. Attempting to re-parse...")
            # Try parsing as regular string first
            try:
                timestamps = pd.to_datetime(df[col_map['timestamp']], format='mixed', errors='coerce')
            except:
                pass

        # Standardize columns
        standardized = pd.DataFrame({
            'platform': 'instagram',
            'username': df[col_map['username']],
            'comment_text': df[col_map['comment']],
            'timestamp': timestamps,
            'likes': df[likes_col] if isinstance(likes_col, str) else likes_col,
            'post_id': self._generate_post_id(csv_path, post_metadata),
            'post_subject': post_metadata.get('subject', '') if post_metadata else '',
            'post_url': post_url,
            'post_caption': post_caption,
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
    
    def save_processed(self, df: pd.DataFrame, filename: str, source_csv_path: str = None):
        """
        Save processed comments to database and/or CSV
        
        Args:
            df: Processed DataFrame
            filename: Output filename
            source_csv_path: Path to source CSV (for tracking)
        """
        # Always save CSV for backup
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} comments to {output_path}")
        
        # Also save to SQLite database if enabled
        if self.use_database and self.engine is not None:
            try:
                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Save to database (append mode, ignore duplicates)
                df.to_sql(
                    'comments',
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                logger.info(f"Saved {len(df)} comments to database")
            except Exception as e:
                logger.warning(f"Could not save to database: {e}")
        
        # Track processed CSV if source path provided
        if source_csv_path:
            self._mark_csv_processed(source_csv_path)
        
        return output_path
    
    def load_all_processed(self) -> pd.DataFrame:
        """Load all processed comment files into single DataFrame"""
        # Try to load from database first (faster)
        if self.use_database and self.engine is not None:
            try:
                df = pd.read_sql("SELECT * FROM comments", self.engine)
                if len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    # Remove duplicates
                    df = df.drop_duplicates(subset=['comment_id'], keep='last')
                    logger.info(f"Loaded {len(df)} comments from database")
                    return df
            except Exception as e:
                logger.warning(f"Could not load from database: {e}. Falling back to CSV files.")
        
        # Fallback to CSV files
        all_files = list(self.processed_dir.glob("*.csv"))
        
        if not all_files:
            return pd.DataFrame()
        
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['comment_id'])
            logger.info(f"Loaded {len(combined)} comments from CSV files")
            return combined
        return pd.DataFrame()
    
    def _init_database(self):
        """Initialize SQLite database and create tables if needed"""
        try:
            self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
            
            # Create table if it doesn't exist
            # Define the table structure
            metadata = MetaData()
            comments_table = Table(
                'comments',
                metadata,
                Column('comment_id', String(50), primary_key=True),
                Column('platform', String(20)),
                Column('username', String(200)),
                Column('comment_text', Text),
                Column('timestamp', DateTime),
                Column('likes', Integer),
                Column('comment_likes', Integer),  # Alias for likes (for clarity)
                Column('post_id', String(50)),
                Column('post_subject', String(200)),
                Column('post_url', String(500)),
                Column('post_caption', Text),  # Add post caption
                Column('primary_emotion', String(50)),
                Column('sentiment_score', Float),
                Column('weighted_sentiment', Float),  # Like-weighted sentiment
                Column('is_sarcastic', Integer),  # SQLite doesn't have boolean
                Column('secondary_emotions', Text),  # JSON string
                Column('mentioned_entities', Text),  # JSON string
            )
            
            # Create table if it doesn't exist
            metadata.create_all(self.engine)
            
        except Exception as e:
            logger.warning(f"Could not initialize database: {e}")
            self.engine = None
            self.use_database = False
    
    def _load_processed_tracker(self) -> Dict:
        """Load the tracker of processed CSV files"""
        if self.processed_tracker_path.exists():
            try:
                with open(self.processed_tracker_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed CSV tracker: {e}")
                return {}
        return {}
    
    def _save_processed_tracker(self):
        """Save the tracker of processed CSV files"""
        try:
            with open(self.processed_tracker_path, 'w') as f:
                json.dump(self._processed_csvs, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save processed CSV tracker: {e}")
    
    def _get_csv_hash(self, csv_path: str) -> str:
        """Generate a unique hash for a CSV file based on path and size"""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            return None
        
        # Use file path + size + modification time for uniqueness
        stat = csv_file.stat()
        unique_str = f"{csv_file.absolute()}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def _mark_csv_processed(self, csv_path: str):
        """Mark a CSV file as processed"""
        csv_file = Path(csv_path)
        csv_hash = self._get_csv_hash(csv_path)
        
        if csv_hash:
            # Store both relative and absolute paths for flexibility
            self._processed_csvs[csv_hash] = {
                'path': str(csv_file.absolute()),
                'relative_path': str(csv_file.relative_to(self.uploads_dir)) if csv_file.is_relative_to(self.uploads_dir) else str(csv_file),
                'processed_at': datetime.now().isoformat(),
                'filename': csv_file.name
            }
            self._save_processed_tracker()
    
    def is_csv_processed(self, csv_path: str) -> bool:
        """
        Check if a CSV file has already been processed
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            True if CSV has been processed
        """
        csv_hash = self._get_csv_hash(csv_path)
        if not csv_hash:
            return False
        
        return csv_hash in self._processed_csvs
    
    def find_unprocessed_csvs(self, platform: Optional[str] = None) -> List[Dict]:
        """
        Find all unprocessed CSV files in the uploads directory
        
        Args:
            platform: Optional filter by platform ('instagram' or 'youtube')
                     If None, tries to auto-detect from filename or content
        
        Returns:
            List of dicts with 'path', 'platform', and 'metadata' for each unprocessed CSV
        """
        unprocessed = []
        
        # Find all CSV files in uploads directory
        csv_files = list(self.uploads_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            # Skip if already processed
            if self.is_csv_processed(str(csv_file)):
                continue
            
            # Try to detect platform
            detected_platform = self._detect_platform_from_file(csv_file, platform)
            
            if detected_platform:
                # Try to extract metadata from file
                metadata = self._extract_metadata_from_csv(csv_file)
                
                unprocessed.append({
                    'path': str(csv_file),
                    'filename': csv_file.name,
                    'platform': detected_platform,
                    'metadata': metadata
                })
        
        return unprocessed
    
    def _detect_platform_from_file(self, csv_file: Path, platform_hint: Optional[str] = None) -> Optional[str]:
        """Detect platform from filename or content"""
        if platform_hint:
            return platform_hint
        
        filename_lower = csv_file.name.lower()
        
        # Check filename for hints
        if 'instagram' in filename_lower or 'ig' in filename_lower:
            return 'instagram'
        if 'youtube' in filename_lower or 'yt' in filename_lower:
            return 'youtube'
        
        # Try to detect from CSV content
        try:
            df = pd.read_csv(csv_file, nrows=5)  # Read first few rows
            columns_lower = [c.lower() for c in df.columns]
            
            # Instagram often has 'Author' column, YouTube has 'Channel Name'
            if any('author' in col for col in columns_lower):
                # Could be either, but check for Instagram-specific patterns
                if any('comment' in col for col in columns_lower):
                    return 'instagram'
            
            if any('channel' in col for col in columns_lower):
                return 'youtube'
        except Exception:
            pass
        
        # Default to instagram if can't determine
        return 'instagram'
    
    def _extract_metadata_from_csv(self, csv_file: Path) -> Dict:
        """
        Try to extract metadata (post_url, subject, etc.) from CSV file
        This handles both regular CSVs and ESUIT-format CSVs
        Auto-preprocesses ESUIT files if detected
        """
        metadata = {}
        processed_file = csv_file  # Default to original file
        
        try:
            # Check if it's ESUIT format and auto-preprocess
            if ESUIT_AVAILABLE and detect_esuit_format(str(csv_file)):
                logger.info(f"Detected ESUIT format, auto-preprocessing: {csv_file.name}")
                try:
                    # Preprocess ESUIT file
                    cleaned_path = preprocess_esuit_csv(str(csv_file))
                    processed_file = Path(cleaned_path)
                    logger.info(f"Preprocessed ESUIT file: {processed_file.name}")
                    
                    # Read the cleaned file to extract metadata
                    df_temp = pd.read_csv(cleaned_path)
                    if 'post_url' in df_temp.columns:
                        metadata['post_url'] = df_temp['post_url'].iloc[0] if len(df_temp) > 0 else ''
                    if 'post_caption' in df_temp.columns:
                        metadata['post_caption'] = df_temp['post_caption'].iloc[0] if len(df_temp) > 0 else ''
                except Exception as e:
                    logger.warning(f"ESUIT preprocessing failed: {e}. Using original file.")
                    processed_file = csv_file
            
            # If not ESUIT or preprocessing failed, try regular extraction
            if not metadata.get('post_url'):
                with open(processed_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if 'instagram.com' in first_line or 'youtube.com' in first_line:
                        metadata['post_url'] = first_line
            
            # Try to extract subject from filename
            filename = csv_file.stem.lower()
            # Common patterns: "taylor_swift_comments.csv", "blake_lively.csv"
            # Remove common suffixes
            for suffix in ['_comments', '_cleaned', '_processed', '_export', '_esuit']:
                filename = filename.replace(suffix, '')
            
            # If filename looks like a name (has underscore or spaces), use as subject
            if '_' in filename or ' ' in filename:
                metadata['subject'] = filename.replace('_', ' ').title()
        
        except Exception as e:
            # If extraction fails, that's okay - metadata is optional
            logger.warning(f"Metadata extraction warning: {e}")
        
        # Store the processed file path for later use
        metadata['_processed_file'] = str(processed_file)
        
        return metadata



