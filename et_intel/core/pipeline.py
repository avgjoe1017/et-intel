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
from .logging_config import get_logger

logger = get_logger(__name__)

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
        logger.info("="*60)
        logger.info("ET SOCIAL INTELLIGENCE PIPELINE")
        logger.info("="*60)
        
        # Step 1: Ingest
        logger.info("STEP 1: Ingesting comments...")
        if platform == 'instagram':
            df = self.ingester.ingest_instagram_csv(csv_path, post_metadata)
        elif platform == 'youtube':
            df = self.ingester.ingest_youtube_csv(csv_path, post_metadata)
        else:
            raise ValueError("Platform must be 'instagram' or 'youtube'")
        
        logger.info(f"Loaded {len(df)} comments from {platform}")
        
        # Step 2: Entity Extraction
        logger.info("STEP 2: Extracting entities...")
        entities = self.entity_extractor.extract_entities_from_comments(df)
        
        logger.info(f"Found {len(entities['people'])} people, {len(entities['shows'])} shows, "
                   f"{len(entities['couples'])} couples/relationships, {len(entities['storylines'])} storylines")
        
        # Save entity knowledge
        self.entity_extractor.save_entity_database(entities)
        
        # Step 3: Sentiment Analysis (with like-weighting)
        logger.info("STEP 3: Analyzing sentiment (with like-weighting)...")
        df = self.sentiment_analyzer.analyze_comments_with_weighting(df)
        
        # Add entity context to each comment
        df = self._tag_comments_with_entities(df, entities)
        
        # Step 4: Save processed data
        logger.info("STEP 4: Saving processed data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{platform}_{timestamp}.csv"
        output_path = self.ingester.save_processed(df, filename, source_csv_path=csv_path)
        
        # Step 5: Save entity analysis
        entity_path = config.PROCESSED_DIR / f"entities_{timestamp}.json"
        with open(entity_path, 'w') as f:
            json.dump(entities, f, indent=2)
        logger.info(f"Saved entity analysis to {entity_path}")
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        return df, entities
    
    def batch_process_unprocessed(self, 
                                   platform: Optional[str] = None,
                                   auto_detect_platform: bool = True) -> Dict:
        """
        Batch process all unprocessed CSV files in the uploads directory
        
        Args:
            platform: Optional platform filter ('instagram' or 'youtube')
                     If None and auto_detect_platform=True, will auto-detect
            auto_detect_platform: If True, tries to detect platform from filename/content
        
        Returns:
            Dict with processing results:
            {
                'total_found': int,
                'processed': int,
                'failed': int,
                'results': List[Dict]  # Individual file results
            }
        """
        logger.info("="*60)
        logger.info("BATCH PROCESSING UNPROCESSED CSVs")
        logger.info("="*60)
        
        # Find unprocessed CSVs
        unprocessed = self.ingester.find_unprocessed_csvs(platform=platform if not auto_detect_platform else None)
        
        if not unprocessed:
            logger.info("No unprocessed CSV files found in uploads directory")
            return {
                'total_found': 0,
                'processed': 0,
                'failed': 0,
                'results': []
            }
        
        logger.info(f"Found {len(unprocessed)} unprocessed CSV file(s)")
        for i, csv_info in enumerate(unprocessed, 1):
            logger.info(f"  {i}. {csv_info['filename']} ({csv_info['platform']})")
        
        results = []
        processed_count = 0
        failed_count = 0
        
        for i, csv_info in enumerate(unprocessed, 1):
            csv_path = csv_info['path']
            detected_platform = csv_info['platform']
            metadata = csv_info.get('metadata', {})
            
            logger.info(f"[{i}/{len(unprocessed)}] Processing: {csv_info['filename']} (Platform: {detected_platform})")
            
            try:
                # Merge provided metadata with auto-detected metadata
                post_metadata = {
                    'subject': metadata.get('subject', ''),
                    'post_url': metadata.get('post_url', ''),
                    'post_caption': metadata.get('post_caption', '')
                }
                
                # Use preprocessed file path if ESUIT was auto-processed
                actual_csv_path = metadata.get('_processed_file', csv_path)
                
                # Process the CSV
                df, entities = self.process_new_data(
                    csv_path=actual_csv_path,
                    platform=detected_platform,
                    post_metadata=post_metadata
                )
                
                results.append({
                    'filename': csv_info['filename'],
                    'path': csv_path,
                    'platform': detected_platform,
                    'status': 'success',
                    'comments_processed': len(df),
                    'entities_found': len(entities['people']) + len(entities['shows'])
                })
                processed_count += 1
                logger.info(f"[OK] Successfully processed {csv_info['filename']}: {len(df)} comments, "
                           f"{len(entities['people']) + len(entities['shows'])} entities")
                
            except Exception as e:
                logger.error(f"[ERROR] Processing {csv_info['filename']}: {e}", exc_info=True)
                results.append({
                    'filename': csv_info['filename'],
                    'path': csv_path,
                    'platform': detected_platform,
                    'status': 'failed',
                    'error': str(e)
                })
                failed_count += 1
        
        logger.info("="*60)
        logger.info(f"BATCH PROCESSING COMPLETE - Total: {len(unprocessed)}, "
                   f"Processed: {processed_count}, Failed: {failed_count}")
        logger.info("="*60)
        
        return {
            'total_found': len(unprocessed),
            'processed': processed_count,
            'failed': failed_count,
            'results': results
        }
    
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
        logger.info("="*60)
        logger.info("GENERATING INTELLIGENCE BRIEF")
        logger.info("="*60)
        
        # Load all processed data
        df = self.ingester.load_all_processed()
        
        if len(df) == 0:
            logger.warning("No processed data found. Please process some CSV files first.")
            return None
        
        logger.info(f"Loaded {len(df)} total comments from database")
        
        # Apply filters
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            logger.info(f"Filtered to {len(df)} comments after {start_date}")
        
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            logger.info(f"Filtered to {len(df)} comments before {end_date}")
        
        if platforms:
            df = df[df['platform'].isin(platforms)]
            logger.info(f"Filtered to {len(df)} comments from {platforms}")
        
        # Extract entities from filtered data
        logger.info("Analyzing entities...")
        entities = self.entity_extractor.extract_entities_from_comments(df)
        
        # Calculate sentiment summaries
        logger.info("Calculating sentiment metrics...")
        sentiment_summary = self._calculate_sentiment_summary(df, entities)
        
        # Calculate velocity for top entities
        logger.info("Calculating sentiment velocity...")
        velocity_alerts = self._calculate_velocity_for_entities(df, entities)
        
        # Demographic insights (basic)
        logger.info("Analyzing demographics...")
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
        
        logger.info(f"Intelligence Brief saved to {brief_path}")
        
        return brief
    
    def _tag_comments_with_entities(self, df: pd.DataFrame, entities: Dict) -> pd.DataFrame:
        """Tag each comment with entities mentioned (explicit and implicit)"""
        people = [p[0] for p in entities['people']]
        shows = [s[0] for s in entities['shows']]
        
        def find_entities(row):
            """Find both explicit and implicit entity mentions"""
            text = str(row.get('comment_text', '')).lower()
            post_subject = str(row.get('post_subject', '')).lower()
            post_caption = str(row.get('post_caption', '')).lower()
            
            # Explicit mentions (name appears in comment)
            mentioned_people = [p for p in people if p.lower() in text]
            mentioned_shows = [s for s in shows if s.lower() in text]
            
            # Implicit mentions (post context - comment is about post subject)
            # Extract entities from post_subject and post_caption
            implicit_people = []
            implicit_shows = []
            
            # Check post_subject for entities
            for person in people:
                if person.lower() in post_subject or person.lower() in post_caption:
                    # If comment doesn't explicitly mention but post is about them,
                    # and comment uses pronouns/relationship terms, count as implicit
                    if person not in mentioned_people:
                        # Check for pronoun/relationship indicators
                        pronoun_indicators = [
                            'they', 'them', 'their', 'this couple', 'these two',
                            'both', 'together', 'relationship', 'dating', 'couple',
                            'she', 'he', 'her', 'him', 'her', 'his'
                        ]
                        if any(indicator in text for indicator in pronoun_indicators):
                            implicit_people.append(person)
            
            for show in shows:
                if show.lower() in post_subject or show.lower() in post_caption:
                    if show not in mentioned_shows:
                        # Check for show-related indicators
                        show_indicators = ['this', 'it', 'the show', 'the movie', 'the series']
                        if any(indicator in text for indicator in show_indicators):
                            implicit_shows.append(show)
            
            # Combine explicit and implicit
            all_people = list(set(mentioned_people + implicit_people))
            all_shows = list(set(mentioned_shows + implicit_shows))
            
            return {
                'people': all_people,
                'shows': all_shows,
                'explicit_people': mentioned_people,
                'implicit_people': implicit_people,
                'explicit_shows': mentioned_shows,
                'implicit_shows': implicit_shows
            }
        
        df['mentioned_entities'] = df.apply(find_entities, axis=1)
        return df
    
    def _calculate_sentiment_summary(self, df: pd.DataFrame, entities: Dict) -> Dict:
        """Calculate sentiment metrics for each entity"""
        summary = {}
        
        # For each person (format: name, total_count, intended, organic, implicit_count)
        for entity_tuple in entities['people']:
            if len(entity_tuple) >= 4:
                person = entity_tuple[0]
                # Handle both old format (4 items) and new format (5 items)
                if len(entity_tuple) == 5:
                    person, total_count, intended, organic, implicit_count = entity_tuple
                else:
                    person, count, intended, organic = entity_tuple
                    implicit_count = 0
            person_comments = df[
                df['comment_text'].str.contains(person, case=False, na=False)
            ]
            
            if len(person_comments) > 0:
                # Calculate weighted sentiment (weighted by likes)
                total_likes = person_comments['likes'].sum() if 'likes' in person_comments.columns else 0
                weighted_avg = None
                if 'weighted_sentiment' in person_comments.columns and total_likes > 0:
                    # Weighted average: sum of weighted_sentiment / sum of likes (normalized)
                    weighted_avg = round(
                        person_comments['weighted_sentiment'].sum() / 
                        (1 + person_comments['likes'].sum()) if total_likes > 0 
                        else person_comments['sentiment_score'].mean(), 
                        3
                    )
                else:
                    # Fallback: calculate from sentiment_score and likes
                    if total_likes > 0:
                        weighted_avg = round(
                            (person_comments['sentiment_score'] * (1 + person_comments['likes'])).sum() / 
                            (1 + total_likes), 
                            3
                        )
                    else:
                        weighted_avg = round(person_comments['sentiment_score'].mean(), 3)
                
                # Find top liked comment
                top_liked_comment = None
                if 'likes' in person_comments.columns and len(person_comments) > 0:
                    top_liked = person_comments.nlargest(1, 'likes')
                    if len(top_liked) > 0 and top_liked['likes'].iloc[0] > 0:
                        top_liked_comment = {
                            'text': top_liked['comment_text'].iloc[0],
                            'likes': int(top_liked['likes'].iloc[0]),
                            'sentiment': round(float(top_liked['sentiment_score'].iloc[0]), 3)
                        }
                
                summary[person] = {
                    'type': 'person',
                    'total_mentions': len(person_comments),
                    'avg_sentiment': round(person_comments['sentiment_score'].mean(), 3),
                    'weighted_avg_sentiment': weighted_avg,
                    'total_likes': int(total_likes),
                    'top_liked_comment': top_liked_comment,
                    'emotion_breakdown': person_comments['primary_emotion'].value_counts().to_dict(),
                    'intended_subject': intended,
                    'organic_mentions': organic
                }
        
        # For each show (format: name, total_count, intended, organic, implicit_count)
        for entity_tuple in entities['shows']:
            if len(entity_tuple) >= 4:
                # Handle both old format (4 items) and new format (5 items)
                if len(entity_tuple) == 5:
                    show, total_count, intended, organic, implicit_count = entity_tuple
                else:
                    show, count, intended, organic = entity_tuple
                    implicit_count = 0
            show_comments = df[
                df['comment_text'].str.contains(show, case=False, na=False)
            ]
            
            if len(show_comments) > 0:
                # Calculate weighted sentiment (weighted by likes)
                total_likes = show_comments['likes'].sum() if 'likes' in show_comments.columns else 0
                weighted_avg = None
                if 'weighted_sentiment' in show_comments.columns and total_likes > 0:
                    weighted_avg = round(
                        show_comments['weighted_sentiment'].sum() / 
                        (1 + show_comments['likes'].sum()) if total_likes > 0 
                        else show_comments['sentiment_score'].mean(), 
                        3
                    )
                else:
                    if total_likes > 0:
                        weighted_avg = round(
                            (show_comments['sentiment_score'] * (1 + show_comments['likes'])).sum() / 
                            (1 + total_likes), 
                            3
                        )
                    else:
                        weighted_avg = round(show_comments['sentiment_score'].mean(), 3)
                
                # Find top liked comment
                top_liked_comment = None
                if 'likes' in show_comments.columns and len(show_comments) > 0:
                    top_liked = show_comments.nlargest(1, 'likes')
                    if len(top_liked) > 0 and top_liked['likes'].iloc[0] > 0:
                        top_liked_comment = {
                            'text': top_liked['comment_text'].iloc[0],
                            'likes': int(top_liked['likes'].iloc[0]),
                            'sentiment': round(float(top_liked['sentiment_score'].iloc[0]), 3)
                        }
                
                summary[show] = {
                    'type': 'show',
                    'total_mentions': len(show_comments),
                    'avg_sentiment': round(show_comments['sentiment_score'].mean(), 3),
                    'weighted_avg_sentiment': weighted_avg,
                    'total_likes': int(total_likes),
                    'top_liked_comment': top_liked_comment,
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
        
        for entity_tuple in top_people:
            if isinstance(entity_tuple, (tuple, list)) and len(entity_tuple) >= 2:
                person = entity_tuple[0]
            else:
                continue
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



