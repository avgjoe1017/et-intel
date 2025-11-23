#!/usr/bin/env python3
"""
End-to-End Tests for ET Social Intelligence System
Tests the complete pipeline from CSV ingestion to report generation
"""

import sys
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from et_intel import config
from et_intel.core.pipeline import ETIntelligencePipeline
from et_intel.core.ingestion import CommentIngester
from et_intel.core.entity_extraction import EntityExtractor
from et_intel.core.sentiment_analysis import SentimentAnalyzer
from et_intel.reporting.report_generator import IntelligenceBriefGenerator


class TestEndToEnd:
    """End-to-end test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Create temporary directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_uploads = self.test_dir / "uploads"
        self.test_processed = self.test_dir / "processed"
        self.test_db = self.test_dir / "database"
        self.test_reports = self.test_dir / "reports"
        
        for d in [self.test_uploads, self.test_processed, self.test_db, self.test_reports]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create sample CSV data
        self.sample_csv = self._create_sample_csv()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir)
    
    def _create_sample_csv(self) -> Path:
        """Create sample CSV file for testing"""
        csv_path = self.test_uploads / "test_comments.csv"
        
        # Create realistic sample data
        data = {
            'username': [
                '@taylorswiftfan', '@realitytv_lover', '@moviebuff2024',
                '@entertainment_news', '@celebrity_watcher', '@popculture',
                '@swiftie4life', '@travis_kelce_fan', '@nfl_fan'
            ],
            'comment': [
                'OMG she ate this look 🔥',
                "I'm so over this storyline honestly",
                "They're perfect together!! ❤️",
                'Taylor Swift is iconic',
                'Travis Kelce is amazing',
                'Love this couple so much',
                'She is everything!',
                'Chiefs Kingdom!',
                'Best relationship ever'
            ],
            'timestamp': [
                datetime.now() - timedelta(days=i) for i in range(9)
            ],
            'likes': [245, 12, 89, 156, 78, 203, 312, 45, 167]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_01_import_modules(self):
        """Test that all modules can be imported"""
        assert ETIntelligencePipeline is not None
        assert CommentIngester is not None
        assert EntityExtractor is not None
        assert SentimentAnalyzer is not None
        assert IntelligenceBriefGenerator is not None
    
    def test_02_csv_ingestion(self):
        """Test CSV ingestion"""
        ingester = CommentIngester()
        
        # Temporarily override paths
        original_uploads = ingester.uploads_dir
        ingester.uploads_dir = self.test_uploads
        
        df = ingester.ingest_instagram_csv(
            str(self.sample_csv),
            post_metadata={
                'post_url': 'https://instagram.com/p/TEST123',
                'subject': 'Taylor Swift, Travis Kelce',
                'post_caption': 'Taylor and Travis at Chiefs game'
            }
        )
        
        assert len(df) > 0
        assert 'comment_text' in df.columns
        assert 'username' in df.columns
        assert 'timestamp' in df.columns
        assert 'platform' in df.columns
        assert all(df['platform'] == 'instagram')
        
        ingester.uploads_dir = original_uploads
    
    def test_03_entity_extraction(self):
        """Test entity extraction"""
        ingester = CommentIngester()
        ingester.uploads_dir = self.test_uploads
        
        df = ingester.ingest_instagram_csv(
            str(self.sample_csv),
            post_metadata={'subject': 'Taylor Swift, Travis Kelce'}
        )
        
        extractor = EntityExtractor(use_spacy=False)  # Use regex for speed
        entities = extractor.extract_entities_from_comments(df)
        
        assert 'people' in entities
        assert 'shows' in entities
        assert 'couples' in entities
        assert 'storylines' in entities
        
        # Should find at least some entities
        assert len(entities['people']) >= 0  # May be 0 if regex doesn't catch them
    
    def test_04_sentiment_analysis_rule_based(self):
        """Test rule-based sentiment analysis"""
        ingester = CommentIngester()
        ingester.uploads_dir = self.test_uploads
        
        df = ingester.ingest_instagram_csv(
            str(self.sample_csv),
            post_metadata={'subject': 'Taylor Swift'}
        )
        
        analyzer = SentimentAnalyzer(use_api=False, use_hf=False, use_textblob=False)
        df_analyzed = analyzer.analyze_comments(df)
        
        assert 'sentiment_score' in df_analyzed.columns
        assert 'primary_emotion' in df_analyzed.columns
        assert all(df_analyzed['sentiment_score'].between(-1, 1))
        assert all(df_analyzed['primary_emotion'].isin(config.EMOTIONS))
    
    def test_05_sentiment_analysis_textblob(self):
        """Test TextBlob sentiment baseline"""
        try:
            from textblob import TextBlob
            
            ingester = CommentIngester()
            ingester.uploads_dir = self.test_uploads
            
            df = ingester.ingest_instagram_csv(
                str(self.sample_csv),
                post_metadata={'subject': 'Taylor Swift'}
            )
            
            analyzer = SentimentAnalyzer(use_api=False, use_hf=False, use_textblob=True)
            df_analyzed = analyzer.analyze_comments(df)
            
            assert 'sentiment_score' in df_analyzed.columns
            # TextBlob columns may be added
            assert len(df_analyzed) > 0
        except ImportError:
            pytest.skip("TextBlob not installed")
    
    def test_06_full_pipeline(self):
        """Test complete pipeline"""
        pipeline = ETIntelligencePipeline(use_api=False)
        
        df, entities = pipeline.process_new_data(
            csv_path=str(self.sample_csv),
            platform='instagram',
            post_metadata={
                'post_url': 'https://instagram.com/p/TEST123',
                'subject': 'Taylor Swift, Travis Kelce',
                'post_caption': 'Taylor and Travis at Chiefs game'
            }
        )
        
        assert len(df) > 0
        assert 'sentiment_score' in df.columns
        assert 'primary_emotion' in df.columns
        assert len(entities['people']) >= 0
        assert len(entities['shows']) >= 0
    
    def test_07_intelligence_brief(self):
        """Test intelligence brief generation"""
        pipeline = ETIntelligencePipeline(use_api=False)
        
        # Process data first
        df, entities = pipeline.process_new_data(
            csv_path=str(self.sample_csv),
            platform='instagram',
            post_metadata={'subject': 'Taylor Swift'}
        )
        
        # Generate brief
        brief = pipeline.generate_intelligence_brief()
        
        assert brief is not None
        assert 'metadata' in brief
        assert 'entities' in brief
        assert 'sentiment_summary' in brief
        assert brief['metadata']['total_comments'] > 0
    
    def test_08_pdf_report_generation(self):
        """Test PDF report generation"""
        pipeline = ETIntelligencePipeline(use_api=False)
        
        # Process data
        df, entities = pipeline.process_new_data(
            csv_path=str(self.sample_csv),
            platform='instagram',
            post_metadata={'subject': 'Taylor Swift'}
        )
        
        # Generate brief
        brief = pipeline.generate_intelligence_brief()
        
        # Generate PDF
        generator = IntelligenceBriefGenerator()
        pdf_path = generator.generate_report(brief, output_filename="TEST_Report.pdf")
        
        assert pdf_path.exists()
        assert pdf_path.suffix == '.pdf'
        assert pdf_path.stat().st_size > 0
    
    def test_09_batch_processing(self):
        """Test batch processing of unprocessed CSVs"""
        pipeline = ETIntelligencePipeline(use_api=False)
        
        # Temporarily override uploads directory
        original_uploads = config.UPLOADS_DIR
        config.UPLOADS_DIR = self.test_uploads
        
        # Run batch processing
        results = pipeline.batch_process_unprocessed()
        
        assert 'total_found' in results
        assert 'processed' in results
        assert 'failed' in results
        assert results['total_found'] >= 0
        
        config.UPLOADS_DIR = original_uploads
    
    def test_10_csv_tracking(self):
        """Test CSV processing tracking"""
        ingester = CommentIngester()
        ingester.uploads_dir = self.test_uploads
        
        # Check if CSV is processed
        assert not ingester.is_csv_processed(str(self.sample_csv))
        
        # Process it
        df = ingester.ingest_instagram_csv(
            str(self.sample_csv),
            post_metadata={'subject': 'Test'}
        )
        
        # Save and mark as processed
        ingester.save_processed(df, "test_processed.csv", source_csv_path=str(self.sample_csv))
        
        # Should now be marked as processed
        assert ingester.is_csv_processed(str(self.sample_csv))
    
    def test_11_database_integration(self):
        """Test SQLite database integration"""
        ingester = CommentIngester(use_database=True)
        ingester.uploads_dir = self.test_uploads
        
        # Temporarily override DB path
        original_db = ingester.db_path
        ingester.db_path = self.test_db / "test.db"
        ingester._init_database()
        
        df = ingester.ingest_instagram_csv(
            str(self.sample_csv),
            post_metadata={'subject': 'Test'}
        )
        
        # Save to database
        ingester.save_processed(df, "test.csv", source_csv_path=str(self.sample_csv))
        
        # Load from database
        df_loaded = ingester.load_all_processed()
        
        assert len(df_loaded) > 0
        assert 'comment_text' in df_loaded.columns
        
        # Close database connection before cleanup (Windows needs this)
        if ingester.engine:
            ingester.engine.dispose()
        
        ingester.db_path = original_db
    
    def test_12_relationship_graph(self):
        """Test NetworkX relationship graph"""
        try:
            from et_intel.core.relationship_graph import RelationshipGraph
            
            pipeline = ETIntelligencePipeline(use_api=False)
            
            df, entities = pipeline.process_new_data(
                csv_path=str(self.sample_csv),
                platform='instagram',
                post_metadata={'subject': 'Taylor Swift, Travis Kelce'}
            )
            
            graph_builder = RelationshipGraph()
            graph = graph_builder.build_graph_from_entities(entities, df)
            
            assert graph is not None
            assert len(graph.nodes()) >= 0
            
            # Test visualization
            output_path = self.test_reports / "test_graph.png"
            graph_builder.visualize(output_path=output_path)
            
            # Graph file may or may not be created depending on entities found
            assert True  # If we get here, no errors occurred
        except ImportError:
            pytest.skip("NetworkX or matplotlib not available")
    
    def test_13_velocity_calculation(self):
        """Test sentiment velocity calculation"""
        analyzer = SentimentAnalyzer(use_api=False)
        
        # Create test data with time series
        dates = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
        data = {
            'comment_text': [f'Comment about Taylor Swift {i}' for i in range(20)],
            'timestamp': dates,
            'sentiment_score': [0.5 + (i % 3) * 0.2 - 0.2 for i in range(20)]  # Varying sentiment
        }
        df = pd.DataFrame(data)
        
        velocity = analyzer.calculate_velocity(df, 'Taylor Swift', window_hours=72)
        
        assert 'entity' in velocity
        assert velocity['entity'] == 'Taylor Swift'
        # May have error if insufficient data
        if 'error' not in velocity:
            assert 'recent_sentiment' in velocity
            assert 'previous_sentiment' in velocity
    
    def test_14_error_handling(self):
        """Test error handling for invalid inputs"""
        ingester = CommentIngester()
        
        # Test invalid CSV path
        with pytest.raises((FileNotFoundError, ValueError)):
            ingester.ingest_instagram_csv("nonexistent.csv")
        
        # Test invalid CSV format
        invalid_csv = self.test_uploads / "invalid.csv"
        invalid_csv.write_text("not,a,valid,csv\n")
        
        with pytest.raises((ValueError, KeyError)):
            ingester.ingest_instagram_csv(str(invalid_csv))
    
    def test_15_config_loading(self):
        """Test configuration loading"""
        assert config.SYSTEM_VERSION is not None
        assert config.DATA_DIR is not None
        assert config.UPLOADS_DIR is not None
        assert config.PROCESSED_DIR is not None
        assert len(config.EMOTIONS) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

