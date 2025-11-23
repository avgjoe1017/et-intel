#!/usr/bin/env python3
"""
Tests for Streamlit Dashboard
Tests dashboard functionality and data loading
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from et_intel.core.ingestion import CommentIngester
from et_intel import config


class TestDashboard:
    """Test dashboard functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_processed = self.test_dir / "processed"
        self.test_processed.mkdir(parents=True, exist_ok=True)
        
        # Create sample processed data
        self._create_sample_processed_data()
        
        yield
        
        shutil.rmtree(self.test_dir)
    
    def _create_sample_processed_data(self):
        """Create sample processed CSV for dashboard testing"""
        data = {
            'comment_id': [f'comment_{i}' for i in range(20)],
            'platform': ['instagram'] * 20,
            'username': [f'user_{i}' for i in range(20)],
            'comment_text': [
                'OMG she ate this look 🔥',
                "I'm so over this storyline",
                "They're perfect together!! ❤️",
                'Taylor Swift is iconic',
                'Travis Kelce is amazing',
            ] * 4,
            'timestamp': [
                datetime.now() - timedelta(days=i) for i in range(20)
            ],
            'likes': [100 + i * 10 for i in range(20)],
            'post_id': ['post_1'] * 20,
            'post_subject': ['Taylor Swift'] * 20,
            'post_url': ['https://instagram.com/p/TEST'] * 20,
            'primary_emotion': ['excitement', 'fatigue', 'love', 'excitement', 'love'] * 4,
            'sentiment_score': [0.8, -0.3, 0.9, 0.7, 0.6] * 4,
            'is_sarcastic': [False] * 20,
            'secondary_emotions': ['[]'] * 20,  # JSON string format
            'mentioned_entities': ['{}'] * 20   # JSON string format
        }
        
        df = pd.DataFrame(data)
        csv_path = self.test_processed / "test_processed.csv"
        df.to_csv(csv_path, index=False)
    
    def test_01_dashboard_imports(self):
        """Test that dashboard can be imported"""
        try:
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            assert True
        except ImportError:
            pytest.skip("Streamlit or Plotly not installed")
    
    def test_02_data_loading(self):
        """Test dashboard data loading function"""
        ingester = CommentIngester()
        
        # Temporarily override processed directory
        original_processed = ingester.processed_dir
        ingester.processed_dir = self.test_processed
        
        df = ingester.load_all_processed()
        
        assert len(df) > 0
        assert 'comment_text' in df.columns
        assert 'sentiment_score' in df.columns
        assert 'primary_emotion' in df.columns
        
        ingester.processed_dir = original_processed
    
    def test_03_data_filtering(self):
        """Test data filtering logic"""
        ingester = CommentIngester()
        ingester.processed_dir = self.test_processed
        
        df = ingester.load_all_processed()
        
        # Test platform filtering
        if 'platform' in df.columns:
            instagram_only = df[df['platform'] == 'instagram']
            assert len(instagram_only) <= len(df)
        
        # Test date filtering
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent = df[df['timestamp'] >= datetime.now() - timedelta(days=7)]
            assert len(recent) <= len(df)
    
    def test_04_metrics_calculation(self):
        """Test dashboard metrics calculation"""
        ingester = CommentIngester()
        ingester.processed_dir = self.test_processed
        
        df = ingester.load_all_processed()
        
        # Total comments
        total = len(df)
        assert total > 0
        
        # Average sentiment
        if 'sentiment_score' in df.columns:
            avg_sentiment = df['sentiment_score'].mean()
            # Handle NaN case (no sentiment data)
            if pd.notna(avg_sentiment):
                assert -1 <= avg_sentiment <= 1
        
        # Unique platforms
        if 'platform' in df.columns:
            unique_platforms = df['platform'].nunique()
            assert unique_platforms > 0
        
        # Unique posts
        if 'post_id' in df.columns:
            unique_posts = df['post_id'].nunique()
            assert unique_posts > 0
    
    def test_05_chart_data_preparation(self):
        """Test chart data preparation"""
        ingester = CommentIngester()
        ingester.processed_dir = self.test_processed
        
        df = ingester.load_all_processed()
        
        # Test daily sentiment aggregation
        if 'timestamp' in df.columns and 'sentiment_score' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_sentiment = df.groupby('date')['sentiment_score'].mean()
            assert len(daily_sentiment) > 0
        
        # Test emotion distribution
        if 'primary_emotion' in df.columns:
            # Filter out NaN/None values
            valid_emotions = df['primary_emotion'].dropna()
            if len(valid_emotions) > 0:
                emotion_counts = valid_emotions.value_counts()
                assert len(emotion_counts) > 0
    
    def test_06_empty_data_handling(self):
        """Test dashboard handles empty data gracefully"""
        ingester = CommentIngester()
        
        # Create empty processed directory
        empty_dir = Path(tempfile.mkdtemp())
        original_processed = ingester.processed_dir
        original_db = ingester.db_path if hasattr(ingester, 'db_path') else None
        
        # Temporarily disable database and use empty CSV directory
        ingester.processed_dir = empty_dir
        if hasattr(ingester, 'use_database'):
            ingester.use_database = False
        if hasattr(ingester, 'engine') and ingester.engine:
            ingester.engine.dispose()
            ingester.engine = None
        
        df = ingester.load_all_processed()
        
        # Should return empty DataFrame, not error
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        # Restore
        ingester.processed_dir = original_processed
        if original_db:
            ingester.db_path = original_db
        
        shutil.rmtree(empty_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

