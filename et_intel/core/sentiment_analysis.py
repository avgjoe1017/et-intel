"""
Sentiment Analysis Pipeline
Context-aware emotion detection using GPT-4o-mini
Handles sarcasm, stan culture, and entertainment-specific language
"""

import pandas as pd
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
    """
    
    def __init__(self, use_api: bool = False):
        """
        Args:
            use_api: If True, uses OpenAI API. If False, uses rule-based fallback
        """
        self.use_api = use_api
        self.emotion_categories = config.EMOTIONS
        
        # For cost tracking
        self.api_calls_made = 0
        self.estimated_cost = 0.0
        
        # Context-aware lexicon for entertainment
        self.sentiment_lexicon = self._build_lexicon()
    
    def analyze_comments(self, df: pd.DataFrame, batch_size: int = None) -> pd.DataFrame:
        """
        Analyze sentiment for all comments in DataFrame
        
        Args:
            df: DataFrame with 'comment_text' column
            batch_size: Process in batches (default from config)
        
        Returns:
            DataFrame with added sentiment columns
        """
        batch_size = batch_size or config.BATCH_SIZE
        
        logger.info(f"Starting sentiment analysis for {len(df)} comments")
        
        results = []
        
        if self.use_api:
            # Use GPT-4o-mini for better accuracy
            results = self._analyze_with_api(df, batch_size)
        else:
            # Use rule-based approach (free, but less accurate)
            results = self._analyze_rule_based(df)
        
        # Add results to dataframe
        df['primary_emotion'] = [r['primary_emotion'] for r in results]
        df['sentiment_score'] = [r['sentiment_score'] for r in results]
        df['secondary_emotions'] = [r.get('secondary_emotions', []) for r in results]
        df['is_sarcastic'] = [r.get('is_sarcastic', False) for r in results]
        
        logger.info(f"Analysis complete. API calls: {self.api_calls_made}, Est. cost: ${self.estimated_cost:.4f}")
        
        return df
    
    def _analyze_with_api(self, df: pd.DataFrame, batch_size: int) -> List[Dict]:
        """Use OpenAI API for sentiment analysis"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
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
        
        for _, row in df.iterrows():
            result = self._analyze_single_comment(row['comment_text'])
            results.append(result)
        
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



