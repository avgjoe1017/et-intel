"""
Main Intelligence Pipeline
Orchestrates ingestion, entity extraction, sentiment analysis, and reporting
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Optional
from .. import config
from .ingestion import CommentIngester
from .entity_extraction import EntityExtractor
from .sentiment_analysis import SentimentAnalyzer

class ETIntelligencePipeline:
    """
    Main orchestrator for the ET Social Intelligence system
    """
    
    def __init__(self, use_api: bool = False):
        """
        Args:
            use_api: Whether to use OpenAI API for sentiment (costs money)
        """
        self.ingester = CommentIngester()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer(use_api=use_api)
        self.use_api = use_api
    
    def process_new_data(self, 
                        csv_path: str, 
                        platform: str,
                        post_metadata: Dict) -> pd.DataFrame:
        """
        Process a new CSV file through the entire pipeline
        
        Args:
            csv_path: Path to CSV file
            platform: 'instagram' or 'youtube'
            post_metadata: Dict with post_url, subject, etc.
        
        Returns:
            Processed DataFrame with all analysis
        """
        print(f"\n{'='*60}")
        print(f"ET SOCIAL INTELLIGENCE PIPELINE")
        print(f"{'='*60}\n")
        
        # Step 1: Ingest
        print("STEP 1: Ingesting comments...")
        if platform == 'instagram':
            df = self.ingester.ingest_instagram_csv(csv_path, post_metadata)
        elif platform == 'youtube':
            df = self.ingester.ingest_youtube_csv(csv_path, post_metadata)
        else:
            raise ValueError("Platform must be 'instagram' or 'youtube'")
        
        print(f"✓ Loaded {len(df)} comments from {platform}")
        
        # Step 2: Entity Extraction
        print("\nSTEP 2: Extracting entities...")
        entities = self.entity_extractor.extract_entities_from_comments(df)
        
        print(f"✓ Found {len(entities['people'])} people")
        print(f"✓ Found {len(entities['shows'])} shows")
        print(f"✓ Found {len(entities['couples'])} couples/relationships")
        print(f"✓ Found {len(entities['storylines'])} storylines")
        
        # Save entity knowledge
        self.entity_extractor.save_entity_database(entities)
        
        # Step 3: Sentiment Analysis
        print("\nSTEP 3: Analyzing sentiment...")
        df = self.sentiment_analyzer.analyze_comments(df)
        
        # Add entity context to each comment
        df = self._tag_comments_with_entities(df, entities)
        
        # Step 4: Save processed data
        print("\nSTEP 4: Saving processed data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{platform}_{timestamp}.csv"
        output_path = self.ingester.save_processed(df, filename)
        
        # Step 5: Save entity analysis
        entity_path = config.PROCESSED_DIR / f"entities_{timestamp}.json"
        with open(entity_path, 'w') as f:
            json.dump(entities, f, indent=2)
        print(f"✓ Saved entity analysis to {entity_path}")
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}\n")
        
        return df, entities
    
    def generate_intelligence_brief(self,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   focus_entities: Optional[list] = None,
                                   platforms: Optional[list] = None) -> Dict:
        """
        Generate Intelligence Brief from processed data
        
        Args:
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
            focus_entities: List of specific entities to focus on
            platforms: List of platforms to include ['instagram', 'youtube']
        
        Returns:
            Dict containing all intelligence data for report generation
        """
        print(f"\n{'='*60}")
        print(f"GENERATING INTELLIGENCE BRIEF")
        print(f"{'='*60}\n")
        
        # Load all processed data
        df = self.ingester.load_all_processed()
        
        if len(df) == 0:
            print("⚠ No processed data found. Please process some CSV files first.")
            return None
        
        print(f"Loaded {len(df)} total comments from database")
        
        # Apply filters
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            print(f"Filtered to {len(df)} comments after {start_date}")
        
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            print(f"Filtered to {len(df)} comments before {end_date}")
        
        if platforms:
            df = df[df['platform'].isin(platforms)]
            print(f"Filtered to {len(df)} comments from {platforms}")
        
        # Extract entities from filtered data
        print("\nAnalyzing entities...")
        entities = self.entity_extractor.extract_entities_from_comments(df)
        
        # Calculate sentiment summaries
        print("Calculating sentiment metrics...")
        sentiment_summary = self._calculate_sentiment_summary(df, entities)
        
        # Calculate velocity for top entities
        print("Calculating sentiment velocity...")
        velocity_alerts = self._calculate_velocity_for_entities(df, entities)
        
        # Demographic insights (basic)
        print("Analyzing demographics...")
        demographics = self._analyze_demographics(df)
        
        # Compile intelligence brief
        brief = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'system_version': config.SYSTEM_VERSION,
                'config_version': config.CONFIG_VERSION,
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'total_comments': len(df),
                'platforms': df['platform'].unique().tolist(),
                'unique_posts': df['post_id'].nunique()
            },
            'entities': entities,
            'sentiment_summary': sentiment_summary,
            'velocity_alerts': velocity_alerts,
            'demographics': demographics,
            'top_storylines': entities['storylines'][:5]
        }
        
        # Save brief
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        brief_path = config.REPORTS_DIR / f"intelligence_brief_{timestamp}.json"
        with open(brief_path, 'w') as f:
            json.dump(brief, f, indent=2)
        
        print(f"\n✓ Intelligence Brief saved to {brief_path}")
        
        return brief
    
    def _tag_comments_with_entities(self, df: pd.DataFrame, entities: Dict) -> pd.DataFrame:
        """Tag each comment with entities mentioned"""
        people = [p[0] for p in entities['people']]
        shows = [s[0] for s in entities['shows']]
        
        def find_entities(text):
            mentioned_people = [p for p in people if p.lower() in text.lower()]
            mentioned_shows = [s for s in shows if s.lower() in text.lower()]
            return {
                'people': mentioned_people,
                'shows': mentioned_shows
            }
        
        df['mentioned_entities'] = df['comment_text'].apply(find_entities)
        return df
    
    def _calculate_sentiment_summary(self, df: pd.DataFrame, entities: Dict) -> Dict:
        """Calculate sentiment metrics for each entity"""
        summary = {}
        
        # For each person
        for person, count, intended, organic in entities['people']:
            person_comments = df[
                df['comment_text'].str.contains(person, case=False, na=False)
            ]
            
            if len(person_comments) > 0:
                summary[person] = {
                    'type': 'person',
                    'total_mentions': len(person_comments),
                    'avg_sentiment': round(person_comments['sentiment_score'].mean(), 3),
                    'emotion_breakdown': person_comments['primary_emotion'].value_counts().to_dict(),
                    'intended_subject': intended,
                    'organic_mentions': organic
                }
        
        # For each show
        for show, count, intended, organic in entities['shows']:
            show_comments = df[
                df['comment_text'].str.contains(show, case=False, na=False)
            ]
            
            if len(show_comments) > 0:
                summary[show] = {
                    'type': 'show',
                    'total_mentions': len(show_comments),
                    'avg_sentiment': round(show_comments['sentiment_score'].mean(), 3),
                    'emotion_breakdown': show_comments['primary_emotion'].value_counts().to_dict(),
                    'intended_subject': intended,
                    'organic_mentions': organic
                }
        
        return summary
    
    def _calculate_velocity_for_entities(self, df: pd.DataFrame, entities: Dict) -> list:
        """Calculate velocity and flag alerts"""
        alerts = []
        
        # Check top 10 entities
        top_people = entities['people'][:10]
        
        for person, count, _, _ in top_people:
            velocity = self.sentiment_analyzer.calculate_velocity(df, person)
            
            if 'error' not in velocity and velocity['alert']:
                alerts.append(velocity)
        
        return sorted(alerts, key=lambda x: abs(x['percent_change']), reverse=True)
    
    def _analyze_demographics(self, df: pd.DataFrame) -> Dict:
        """Basic demographic analysis from available data"""
        # Platform breakdown
        platform_dist = df['platform'].value_counts().to_dict()
        
        # Temporal patterns
        df['hour'] = df['timestamp'].dt.hour
        hour_dist = df['hour'].value_counts().sort_index().to_dict()
        
        # Engagement patterns
        high_engagement = df[df['likes'] > df['likes'].quantile(0.75)]
        
        return {
            'platform_distribution': platform_dist,
            'peak_hours': hour_dist,
            'high_engagement_sentiment': round(high_engagement['sentiment_score'].mean(), 3),
            'low_engagement_sentiment': round(df[df['likes'] <= df['likes'].quantile(0.25)]['sentiment_score'].mean(), 3)
        }



