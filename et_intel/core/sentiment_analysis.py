"""
Sentiment Analysis Pipeline
Context-aware emotion detection using GPT-4o-mini
Handles sarcasm, stan culture, and entertainment-specific language
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import json
import time
from collections import defaultdict
import logging
from .. import config

# Set up logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes sentiment with entertainment context awareness
    Supports multiple methods: rule-based, TextBlob, Hugging Face, OpenAI API
    """
    
    def __init__(self, use_api: bool = False, use_hf: bool = True, use_textblob: bool = True):
        """
        Args:
            use_api: If True, uses OpenAI API. If False, uses rule-based fallback
            use_hf: If True, uses Hugging Face emotion classifier (free, accurate)
            use_textblob: If True, uses TextBlob for sentiment baseline
        """
        self.use_api = use_api
        self.use_hf = use_hf
        self.use_textblob = use_textblob
        self.emotion_categories = config.EMOTIONS
        
        # For cost tracking
        self.api_calls_made = 0
        self.estimated_cost = 0.0
        
        # Context-aware lexicon for entertainment
        self.sentiment_lexicon = self._build_lexicon()
        
        # Initialize Hugging Face emotion classifier (lazy loading)
        self.hf_classifier = None
        if self.use_hf:
            self._init_hf_classifier()
        
        # Initialize TextBlob (lazy loading)
        self.textblob_available = False
        if self.use_textblob:
            self._init_textblob()
    
    def analyze_comments(self, df: pd.DataFrame, batch_size: int = None) -> pd.DataFrame:
        """
        Analyze sentiment for all comments in DataFrame
        
        Args:
            df: DataFrame with 'comment_text' column
            batch_size: Process in batches (default from config)
        
        Returns:
            DataFrame with added sentiment columns
        """
        try:
            # Validate input
            if df.empty:
                logger.warning("Empty DataFrame provided to analyze_comments")
                return df
            
            if 'comment_text' not in df.columns:
                logger.error("DataFrame missing 'comment_text' column")
                # Try to find alternative column names
                for col in ['comment', 'text', 'content', 'message']:
                    if col in df.columns:
                        df['comment_text'] = df[col]
                        logger.info(f"Using '{col}' column as comment_text")
                        break
                else:
                    raise ValueError("No comment text column found in DataFrame")
            
            # Clean bad data
            df = df.copy()
            original_len = len(df)
            df = df[df['comment_text'].notna() & (df['comment_text'].astype(str).str.strip() != '')]
            if len(df) < original_len:
                logger.warning(f"Filtered out {original_len - len(df)} empty/invalid comments")
            
            batch_size = batch_size or config.BATCH_SIZE
            
            logger.info(f"Starting sentiment analysis for {len(df)} comments")
            
            results = []
            
            try:
                if self.use_api:
                    # Check if API key is available
                    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.strip() == "":
                        logger.warning("OpenAI API key not found. Falling back to rule-based sentiment analysis.")
                        logger.info("To use OpenAI API, set OPENAI_API_KEY in .env file or use --no-api flag for free analysis.")
                        results = self._analyze_rule_based(df)
                    else:
                        # Use GPT-4o-mini for better accuracy
                        results = self._analyze_with_api(df, batch_size)
                elif self.use_hf and self.hf_classifier:
                    # Use Hugging Face emotion classifier (free, accurate)
                    results = self._analyze_with_hf(df)
                else:
                    # Use rule-based approach (free, but less accurate)
                    results = self._analyze_rule_based(df)
            except Exception as e:
                logger.error(f"Error in sentiment analysis method: {e}. Falling back to rule-based.")
                results = self._analyze_rule_based(df)
            
            # Add TextBlob baseline if available (for comparison)
            if self.use_textblob and self.textblob_available:
                try:
                    results = self._add_textblob_baseline(df, results)
                except Exception as e:
                    logger.warning(f"TextBlob baseline failed: {e}. Continuing without it.")
            
            # Ensure we have results for all rows
            if len(results) < len(df):
                logger.warning(f"Only {len(results)} results for {len(df)} comments. Padding with defaults.")
                while len(results) < len(df):
                    results.append({
                        'primary_emotion': 'neutral',
                        'sentiment_score': 0.0,
                        'secondary_emotions': [],
                        'is_sarcastic': False
                    })
            
            # Add results to dataframe
            df['primary_emotion'] = [r.get('primary_emotion', 'neutral') for r in results[:len(df)]]
            df['sentiment_score'] = [r.get('sentiment_score', 0.0) for r in results[:len(df)]]
            df['secondary_emotions'] = [r.get('secondary_emotions', []) for r in results[:len(df)]]
            df['is_sarcastic'] = [r.get('is_sarcastic', False) for r in results[:len(df)]]
            
            logger.info(f"Analysis complete. API calls: {self.api_calls_made}, Est. cost: ${self.estimated_cost:.4f}")
            
        except Exception as e:
            logger.error(f"Critical error in analyze_comments: {e}")
            # Return dataframe with default values
            df['primary_emotion'] = 'neutral'
            df['sentiment_score'] = 0.0
            df['secondary_emotions'] = [[]] * len(df)
            df['is_sarcastic'] = False
        
        return df
    
    def analyze_comments_with_weighting(self, df: pd.DataFrame, batch_size: int = None) -> pd.DataFrame:
        """
        Analyze sentiment with like-weighting
        
        Formula: sentiment_score * (1 + log(1 + likes))
        This gives more weight to highly-liked comments without letting one viral comment dominate
        
        Args:
            df: DataFrame with 'comment_text' and optionally 'likes' column
            batch_size: Process in batches (default from config)
        
        Returns:
            DataFrame with added sentiment columns including 'weighted_sentiment'
        """
        # Regular sentiment analysis first
        df = self.analyze_comments(df, batch_size)
        
        # Ensure 'likes' column exists (default to 0 if missing)
        if 'likes' not in df.columns:
            df['likes'] = 0
            logger.warning("'likes' column not found. Using 0 for all comments.")
        
        # Ensure likes are numeric
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
        
        # Calculate weighted sentiment
        # Formula: sentiment_score * (1 + log(1 + likes))
        # This gives more weight to highly-liked comments without letting one viral comment dominate
        # Using log prevents a single comment with 10,000 likes from completely dominating
        df['weighted_sentiment'] = df['sentiment_score'] * (1 + np.log1p(df['likes']))
        
        # Also add comment_likes column (alias for clarity in database)
        df['comment_likes'] = df['likes']
        
        logger.info(f"Weighted sentiment calculated. Max likes: {df['likes'].max()}, "
                   f"Avg weighted sentiment: {df['weighted_sentiment'].mean():.3f}")
        
        return df
    
    def _analyze_with_api(self, df: pd.DataFrame, batch_size: int) -> List[Dict]:
        """Use OpenAI API for sentiment analysis"""
        # Check if API key is available
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.strip() == "" or config.OPENAI_API_KEY == "your-openai-api-key-here":
            logger.warning("OpenAI API key not configured. Falling back to rule-based analysis.")
            logger.info("To use OpenAI API: 1) Create .env file, 2) Add OPENAI_API_KEY=your-key, 3) Or use --no-api flag for free analysis.")
            return self._analyze_rule_based(df)
        
        try:
            from openai import OpenAI
            # Strip whitespace from API key (in case .env file has trailing spaces)
            api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else ""
            if not api_key:
                logger.warning("OpenAI API key is empty. Falling back to rule-based analysis.")
                return self._analyze_rule_based(df)
            client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.warning(f"API not available: {e}. Falling back to rule-based analysis")
            return self._analyze_rule_based(df)
        
        results = []
        max_retries = 3
        
        # Process in batches to manage costs
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Create batch prompt
            comments_text = "\n".join([
                f"{idx}. {row['comment_text'][:200]}" 
                for idx, row in batch.iterrows()
            ])
            
            prompt = f"""Analyze these social media comments from Entertainment Tonight's audience. 
For each comment, identify:
1. Primary emotion: {', '.join(self.emotion_categories)}
2. Sentiment score: -1 (very negative) to +1 (very positive)
3. Is it sarcastic? (yes/no)

IMPORTANT CONTEXT:
- "I can't even" = EXCITEMENT (positive)
- "She ate" / "He ate" = LOVE/EXCITEMENT (very positive)
- "This is everything" = EXCITEMENT (positive)
- "Not [person] doing [thing]" = can be positive or negative based on context
- "Living for this" = EXCITEMENT (positive)
- Excessive caps + exclamation = usually EXCITEMENT

Comments:
{comments_text}

Respond in JSON format:
[
  {{"index": 0, "emotion": "excitement", "score": 0.8, "sarcastic": false}},
  ...
]"""
            
            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=config.SENTIMENT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert at analyzing social media sentiment in entertainment contexts."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=config.MAX_TOKENS_SENTIMENT * len(batch),
                        timeout=30.0  # Add timeout
                    )
                    
                    self.api_calls_made += 1
                    
                    # Estimate cost
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    self.estimated_cost += (input_tokens / 1_000_000 * 0.15) + (output_tokens / 1_000_000 * 0.60)
                    
                    # Parse response with JSON repair
                    response_text = response.choices[0].message.content
                    batch_results = self._parse_json_with_repair(response_text)
                    
                    # Map results back to dataframe indices
                    for item in batch_results:
                        results.append({
                            'primary_emotion': item.get('emotion', 'neutral'),
                            'sentiment_score': float(item.get('score', 0.0)),
                            'is_sarcastic': item.get('sarcastic', False),
                            'secondary_emotions': []
                        })
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    logger.warning(f"API error on batch {i}, attempt {attempt + 1}/{max_retries}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Max retries exceeded - fall back to rule-based for this batch
                        logger.error(f"Max retries exceeded for batch {i}. Using rule-based fallback")
                        for _, row in batch.iterrows():
                            results.append(self._analyze_single_comment(row['comment_text']))
            
            # Rate limiting (be nice to the API)
            time.sleep(0.5)
        
        return results
    
    def _parse_json_with_repair(self, response_text: str) -> List[Dict]:
        """Parse JSON with automatic repair attempts"""
        # Try direct parse
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        if "```json" in response_text:
            try:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Try extracting from generic code blocks
        if "```" in response_text:
            try:
                json_str = response_text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Try finding JSON array boundaries
        try:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If all repair attempts fail, log and return empty
        logger.error(f"Could not parse JSON response: {response_text[:200]}...")
        return []
    
    def _analyze_rule_based(self, df: pd.DataFrame) -> List[Dict]:
        """Rule-based sentiment analysis (free fallback)"""
        results = []
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                comment_text = str(row.get('comment_text', '')) if pd.notna(row.get('comment_text')) else ''
                if comment_text:
                    result = self._analyze_single_comment(comment_text)
                    results.append(result)
                else:
                    results.append({
                        'primary_emotion': 'neutral',
                        'sentiment_score': 0.0,
                        'is_sarcastic': False,
                        'secondary_emotions': []
                    })
            except Exception as e:
                failed_count += 1
                logger.debug(f"Error in rule-based analysis for comment {idx}: {e}")
                results.append({
                    'primary_emotion': 'neutral',
                    'sentiment_score': 0.0,
                    'is_sarcastic': False,
                    'secondary_emotions': []
                })
        
        if failed_count > 0:
            logger.warning(f"Failed to analyze {failed_count} comments with rule-based method")
        
        return results
    
    def _analyze_single_comment(self, text: str) -> Dict:
        """Analyze single comment using rule-based approach"""
        text_lower = text.lower()
        
        # Initialize scores
        emotion_scores = {emotion: 0 for emotion in self.emotion_categories}
        
        # Check for excitement indicators
        if any(phrase in text_lower for phrase in self.sentiment_lexicon['excitement']):
            emotion_scores['excitement'] += 2
        
        # Check for love/positive
        if any(phrase in text_lower for phrase in self.sentiment_lexicon['love']):
            emotion_scores['love'] += 2
        
        # Check for anger
        if any(phrase in text_lower for phrase in self.sentiment_lexicon['anger']):
            emotion_scores['anger'] += 2
        
        # Check for disappointment
        if any(phrase in text_lower for phrase in self.sentiment_lexicon['disappointment']):
            emotion_scores['disappointment'] += 2
        
        # Check for disgust
        if any(phrase in text_lower for phrase in self.sentiment_lexicon['disgust']):
            emotion_scores['disgust'] += 2
        
        # Check for fatigue (storyline exhaustion)
        if any(phrase in text_lower for phrase in self.sentiment_lexicon['fatigue']):
            emotion_scores['fatigue'] += 2
        
        # Caps + exclamation = excitement (unless angry words present)
        if text.isupper() or text.count('!') >= 2:
            if emotion_scores['anger'] == 0:
                emotion_scores['excitement'] += 1
        
        # Emojis
        excitement_emojis = ['🔥', '😍', '💕', '❤️', '🥰', '✨', '👏', '🙌']
        if any(emoji in text for emoji in excitement_emojis):
            emotion_scores['excitement'] += 1
        
        # Determine primary emotion
        if sum(emotion_scores.values()) == 0:
            primary = 'neutral'
            score = 0.0
        else:
            primary = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate sentiment score
            positive_emotions = ['excitement', 'love', 'surprise']
            negative_emotions = ['anger', 'disappointment', 'disgust', 'fatigue']
            
            positive_score = sum(emotion_scores[e] for e in positive_emotions)
            negative_score = sum(emotion_scores[e] for e in negative_emotions)
            
            total = positive_score + negative_score
            if total > 0:
                score = (positive_score - negative_score) / total
            else:
                score = 0.0
        
        # Sarcasm detection (basic)
        is_sarcastic = any(indicator in text_lower for indicator in [
            'yeah right', 'sure jan', 'totally', 'definitely', '/s'
        ])
        
        return {
            'primary_emotion': primary,
            'sentiment_score': score,
            'is_sarcastic': is_sarcastic,
            'secondary_emotions': [e for e, s in emotion_scores.items() if s > 0 and e != primary]
        }
    
    def _build_lexicon(self) -> Dict[str, List[str]]:
        """Build entertainment-aware sentiment lexicon"""
        return {
            'excitement': [
                "i can't even", "she ate", "he ate", "this is everything",
                "living for", "obsessed", "iconic", "legend", "queen", "king",
                "amazing", "love this", "so good", "perfect", "yass", "yasss",
                "here for it", "we stan", "omg", "🔥"
            ],
            'love': [
                "love", "adorable", "cute", "sweet", "beautiful", "gorgeous",
                "stunning", "perfect together", "ship", "goals", "heart",
                "aww", "precious", "angel"
            ],
            'anger': [
                "angry", "mad", "furious", "pissed", "hate", "terrible",
                "awful", "disgusting", "worst", "wtf", "seriously",
                "unacceptable", "ridiculous", "trash", "garbage"
            ],
            'disappointment': [
                "disappointed", "let down", "expected better", "thought",
                "unfortunately", "sad", "unfortunate", "wished", "hoped",
                "could have been", "should have"
            ],
            'disgust': [
                "gross", "ew", "yuck", "disgusting", "nasty", "sick",
                "vile", "revolting", "creepy", "cringe"
            ],
            'fatigue': [
                "again", "still", "move on", "over it", "tired of",
                "enough already", "next", "boring", "old news", "done with",
                "sick of hearing", "can we stop"
            ],
            'surprise': [
                "wow", "wait what", "no way", "really", "shocked",
                "unexpected", "didn't see", "plot twist", "omg"
            ]
        }
    
    def _init_hf_classifier(self):
        """
        Initialize Hugging Face emotion classifier (lazy loading)
        
        IMPORTANT: Model version is pinned to prevent drift.
        If you need to update, explicitly change the revision.
        """
        try:
            from transformers import pipeline
            # Pin model version to prevent unexpected changes
            # Model: j-hartmann/emotion-english-distilroberta-base
            # Revision: main (explicitly set for reproducibility)
            MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
            MODEL_REVISION = "main"  # Pin to specific commit for production
            
            logger.info(f"Loading Hugging Face emotion classifier: {MODEL_NAME} (revision: {MODEL_REVISION})")
            self.hf_classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                revision=MODEL_REVISION,  # Pin revision to prevent drift
                return_all_scores=True,
                device=-1  # Use CPU
            )
            logger.info("✓ Hugging Face classifier loaded")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face classifier: {e}")
            self.hf_classifier = None
            self.use_hf = False
    
    def _init_textblob(self):
        """Initialize TextBlob (lazy loading)"""
        try:
            from textblob import TextBlob
            self.textblob_available = True
            logger.info("✓ TextBlob available")
        except Exception as e:
            logger.warning(f"TextBlob not available: {e}")
            self.textblob_available = False
    
    def _analyze_with_hf(self, df: pd.DataFrame) -> List[Dict]:
        """Use Hugging Face emotion classifier for sentiment analysis"""
        if not self.hf_classifier:
            logger.warning("HF classifier not available, falling back to rule-based")
            return self._analyze_rule_based(df)
        
        results = []
        logger.info(f"Analyzing {len(df)} comments with Hugging Face emotion classifier")
        
        # Emotion mapping from HF model to our categories
        hf_to_our_emotions = {
            'joy': 'excitement',
            'love': 'love',
            'anger': 'anger',
            'sadness': 'disappointment',
            'fear': 'surprise',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                comment = str(row['comment_text'])[:512]  # Limit length for HF model
                hf_results = self.hf_classifier(comment)
                
                # Get top emotion
                top_emotion = max(hf_results[0], key=lambda x: x['score'])
                hf_label = top_emotion['label'].lower()
                hf_score = top_emotion['score']
                
                # Map to our emotion categories
                primary_emotion = hf_to_our_emotions.get(hf_label, 'neutral')
                
                # Calculate sentiment score from emotion probabilities
                positive_scores = sum(
                    item['score'] for item in hf_results[0] 
                    if item['label'].lower() in ['joy', 'love']
                )
                negative_scores = sum(
                    item['score'] for item in hf_results[0]
                    if item['label'].lower() in ['anger', 'sadness', 'fear']
                )
                
                sentiment_score = (positive_scores - negative_scores)
                
                # Get secondary emotions
                sorted_emotions = sorted(hf_results[0], key=lambda x: x['score'], reverse=True)
                secondary = [
                    hf_to_our_emotions.get(item['label'].lower(), 'neutral')
                    for item in sorted_emotions[1:3]  # Top 2-3
                    if item['score'] > 0.1 and hf_to_our_emotions.get(item['label'].lower()) != primary_emotion
                ]
                
                results.append({
                    'primary_emotion': primary_emotion,
                    'sentiment_score': round(sentiment_score, 3),
                    'is_sarcastic': False,  # HF doesn't detect sarcasm
                    'secondary_emotions': secondary,
                    'hf_confidence': round(hf_score, 3)
                })
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error analyzing comment {idx} with HF: {e}")
                # Fallback to rule-based for this comment
                try:
                    comment_text = str(row.get('comment_text', '')) if pd.notna(row.get('comment_text')) else ''
                    if comment_text:
                        results.append(self._analyze_single_comment(comment_text))
                    else:
                        results.append({
                            'primary_emotion': 'neutral',
                            'sentiment_score': 0.0,
                            'is_sarcastic': False,
                            'secondary_emotions': []
                        })
                except Exception as e2:
                    logger.error(f"Fallback analysis also failed for comment {idx}: {e2}")
                    results.append({
                        'primary_emotion': 'neutral',
                        'sentiment_score': 0.0,
                        'is_sarcastic': False,
                        'secondary_emotions': []
                    })
        
        if failed_count > 0:
            logger.warning(f"Failed to analyze {failed_count} comments with HF, used fallback")
        
        logger.info(f"✓ HF analysis complete for {len(results)} comments")
        return results
    
    def _add_textblob_baseline(self, df: pd.DataFrame, results: List[Dict]) -> List[Dict]:
        """Add TextBlob sentiment baseline for comparison"""
        try:
            from textblob import TextBlob
            
            for i, (_, row) in enumerate(df.iterrows()):
                if i < len(results):
                    try:
                        blob = TextBlob(str(row['comment_text']))
                        polarity = blob.sentiment.polarity  # -1 to +1
                        subjectivity = blob.sentiment.subjectivity  # 0 to 1
                        
                        # Add TextBlob scores to results
                        results[i]['textblob_polarity'] = round(polarity, 3)
                        results[i]['textblob_subjectivity'] = round(subjectivity, 3)
                        
                        # Ensemble: average our score with TextBlob
                        if 'sentiment_score' in results[i]:
                            ensemble_score = (results[i]['sentiment_score'] + polarity) / 2
                            results[i]['sentiment_score_ensemble'] = round(ensemble_score, 3)
                    except Exception as e:
                        logger.debug(f"TextBlob error for comment {i}: {e}")
                        results[i]['textblob_polarity'] = None
                        results[i]['textblob_subjectivity'] = None
        except Exception as e:
            logger.warning(f"TextBlob not available: {e}")
        
        return results
    
    def calculate_velocity(self, df: pd.DataFrame, entity: str, window_hours: int = None) -> Dict:
        """
        Calculate sentiment velocity for an entity
        
        Args:
            df: DataFrame with sentiment data
            entity: Name of person/show to track
            window_hours: Time window (default from config)
        
        Returns:
            Dict with velocity metrics
        """
        window_hours = window_hours or config.VELOCITY_WINDOW_HOURS
        min_sample = config.MIN_VELOCITY_SAMPLE_SIZE
        
        # Filter comments mentioning entity
        entity_comments = df[df['comment_text'].str.contains(entity, case=False, na=False)]
        
        if len(entity_comments) == 0:
            logger.warning(f"No comments found for entity: {entity}")
            return {'error': 'No comments found for entity', 'entity': entity}
        
        # Sort by timestamp
        entity_comments = entity_comments.sort_values('timestamp')
        
        # Split into recent window vs. previous period
        latest_time = entity_comments['timestamp'].max()
        window_start = latest_time - pd.Timedelta(hours=window_hours)
        
        recent = entity_comments[entity_comments['timestamp'] >= window_start]
        previous = entity_comments[entity_comments['timestamp'] < window_start]
        
        # Validate sample sizes
        if len(recent) < min_sample:
            logger.warning(f"Insufficient recent data for {entity}: {len(recent)} < {min_sample}")
            return {
                'error': f'Insufficient recent data (need {min_sample}, have {len(recent)})',
                'entity': entity
            }
        
        if len(previous) < min_sample:
            logger.warning(f"Insufficient historical data for {entity}: {len(previous)} < {min_sample}")
            return {
                'error': f'Insufficient historical data (need {min_sample}, have {len(previous)})',
                'entity': entity
            }
        
        recent_sentiment = recent['sentiment_score'].mean()
        previous_sentiment = previous['sentiment_score'].mean()
        
        # Calculate change
        sentiment_change = recent_sentiment - previous_sentiment
        percent_change = (sentiment_change / abs(previous_sentiment)) * 100 if previous_sentiment != 0 else 0
        
        # Determine if alert threshold crossed
        alert = abs(percent_change) >= (config.VELOCITY_ALERT_THRESHOLD * 100)
        
        result = {
            'entity': entity,
            'recent_sentiment': round(recent_sentiment, 3),
            'previous_sentiment': round(previous_sentiment, 3),
            'sentiment_change': round(sentiment_change, 3),
            'percent_change': round(percent_change, 1),
            'alert': alert,
            'recent_comment_count': len(recent),
            'previous_comment_count': len(previous),
            'window_hours': window_hours,
            'window_start': window_start.isoformat(),
            'window_end': latest_time.isoformat()
        }
        
        if alert:
            logger.warning(f"VELOCITY ALERT: {entity} changed {percent_change:+.1f}% in {window_hours}hrs")
        
        return result



