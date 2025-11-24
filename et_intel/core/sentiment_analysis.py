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
        
        # Initialize OpenAI client if API is enabled and key is available
        self.openai_client = None
        if self.use_api and config.OPENAI_API_KEY and config.OPENAI_API_KEY.strip() and config.OPENAI_API_KEY != "your-openai-api-key-here":
            try:
                from openai import OpenAI
                import os
                # Set as environment variable (OpenAI library prefers this)
                api_key = config.OPENAI_API_KEY.strip()
                os.environ['OPENAI_API_KEY'] = api_key
                # Also initialize client with explicit key
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully with API key")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self.openai_client = None
    
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
                    elif config.SENTIMENT_USE_BATCH_API:
                        # Batch API mode: 50% cheaper, asynchronous processing
                        results = self._analyze_with_batch_api(df)
                    elif config.SENTIMENT_USE_HYBRID:
                        # Hybrid mode: nano first, escalate important comments to 4o-mini
                        results = self._analyze_with_api_hybrid(df, batch_size)
                    else:
                        # Standard mode: Use GPT-4o-mini for all comments
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
        
        # Get API key
        import os
        api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else ""
        if not api_key:
            logger.warning("OpenAI API key is empty. Falling back to rule-based analysis.")
            return self._analyze_rule_based(df)
        
        # Set environment variable (OpenAI library checks this)
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Use pre-initialized client if available, otherwise create new one
        if self.openai_client is not None:
            client = self.openai_client
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                logger.info("Created new OpenAI client for API analysis")
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
                    # Ensure API key is available
                    import os
                    api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else ""
                    if not api_key:
                        logger.error("API key is missing. Cannot make API call.")
                        break
                    
                    # Set environment variable (OpenAI library checks this)
                    os.environ['OPENAI_API_KEY'] = api_key
                    
                    # Use existing client if available and valid, otherwise create new one
                    if client is None or not hasattr(client, '_client'):
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        logger.debug(f"Created OpenAI client for batch {i+1}")
                    else:
                        # Ensure the existing client has the API key
                        if hasattr(client, '_client') and hasattr(client._client, 'api_key'):
                            if client._client.api_key != api_key:
                                # Recreate if key changed
                                from openai import OpenAI
                                client = OpenAI(api_key=api_key)
                                logger.debug(f"Recreated OpenAI client (key changed)")
                    
                    response = client.chat.completions.create(
                        model=config.SENTIMENT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert at analyzing social media sentiment in entertainment contexts."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=config.MAX_TOKENS_SENTIMENT * len(batch),
                        timeout=90.0  # Add timeout (increased to 90s for stability)
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
                    error_msg = str(e)
                    logger.warning(f"API error on batch {i}, attempt {attempt + 1}/{max_retries}: {error_msg}")
                    
                    # Check if it's an authentication error
                    if "authentication" in error_msg.lower() or "bearer" in error_msg.lower() or "api key" in error_msg.lower():
                        # Recreate client with fresh API key
                        import os
                        api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else ""
                        if api_key:
                            os.environ['OPENAI_API_KEY'] = api_key
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            logger.info(f"Recreated OpenAI client due to authentication error (attempt {attempt + 1})")
                    
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
            logger.info("[OK] Hugging Face classifier loaded")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face classifier: {e}")
            self.hf_classifier = None
            self.use_hf = False
    
    def _init_textblob(self):
        """Initialize TextBlob (lazy loading)"""
        try:
            from textblob import TextBlob
            self.textblob_available = True
            logger.info("[OK] TextBlob available")
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
        
        logger.info(f"[OK] HF analysis complete for {len(results)} comments")
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
    
    def _call_openai_model_batch(
        self,
        client,
        model_name: str,
        comments: List[str],
        system_prompt: str,
    ) -> List[Dict]:
        """
        Call a given OpenAI chat model for a batch of comments.
        Returns list of dicts with emotion/score/sarcasm/confidence.
        
        This is a generic helper used by both standard and hybrid analysis paths.
        """
        import json
        import os
        
        api_key = config.OPENAI_API_KEY.strip() if config.OPENAI_API_KEY else ""
        if not api_key:
            logger.warning("OPENAI_API_KEY missing in _call_openai_model_batch; using rule-based fallback")
            return [self._analyze_single_comment(c) for c in comments]
        
        os.environ["OPENAI_API_KEY"] = api_key
        
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        
        # Build prompt
        numbered = "\n".join(f"{i}. {c[:200]}" for i, c in enumerate(comments))
        prompt = f"""{system_prompt}

Comments:
{numbered}

Respond in strict JSON array format:
[
  {{"index": 0, "emotion": "excitement", "score": 0.8, "sarcastic": false, "confidence": 0.9}},
  {{"index": 1, "emotion": "anger", "score": -0.6, "sarcastic": false, "confidence": 0.8}},
  ...
]"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing social media sentiment in entertainment contexts. Always return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=config.MAX_TOKENS_SENTIMENT * max(1, len(comments)),
                timeout=60.0,
            )
            
            self.api_calls_made += 1
            
            # Cost tracking
            try:
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                # Model-specific pricing
                if "gpt-5-nano" in model_name.lower():
                    # gpt-5-nano: $0.05/$0.40 per 1M tokens
                    input_cost = input_tokens / 1_000_000 * 0.05
                    output_cost = output_tokens / 1_000_000 * 0.40
                elif "gpt-4o-mini" in model_name.lower():
                    # gpt-4o-mini: $0.15/$0.60 per 1M tokens
                    input_cost = input_tokens / 1_000_000 * 0.15
                    output_cost = output_tokens / 1_000_000 * 0.60
                else:
                    # Default to gpt-4o-mini pricing for unknown models
                    input_cost = input_tokens / 1_000_000 * 0.15
                    output_cost = output_tokens / 1_000_000 * 0.60
                
                self.estimated_cost += input_cost + output_cost
            except Exception as e:
                logger.debug(f"No usage info available: {e}")
            
            raw = response.choices[0].message.content
            items = self._parse_json_with_repair(raw)
            
            results = []
            for item in items:
                results.append({
                    "primary_emotion": item.get("emotion", "neutral"),
                    "sentiment_score": float(item.get("score", 0.0)),
                    "is_sarcastic": bool(item.get("sarcastic", False)),
                    "secondary_emotions": [],
                    "model": model_name,
                    "model_confidence": float(item.get("confidence", abs(float(item.get("score", 0.0))))),
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"OpenAI API call failed for model {model_name}: {e}. Using rule-based fallback.")
            return [self._analyze_single_comment(c) for c in comments]
    
    def _analyze_with_api_hybrid(self, df: pd.DataFrame, batch_size: int) -> List[Dict]:
        """
        Hybrid pipeline:
          1) Run cheap model (nano) on all comments (fast, cheap).
          2) Identify 'escalation' comments that need high-accuracy model.
          3) Re-run only those through gpt-4o-mini and overwrite.
        
        This provides cost optimization while maintaining quality on important comments.
        """
        import math
        
        if df.empty:
            return []
        
        comments = df["comment_text"].astype(str).tolist()
        
        # Ensure OpenAI key
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.strip() in ("", "your-openai-api-key-here"):
            logger.warning("OPENAI_API_KEY missing; hybrid mode falling back to rule-based sentiment")
            return self._analyze_rule_based(df)
        
        # Initialize client once
        client = self.openai_client
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY.strip())
                self.openai_client = client
            except Exception as e:
                logger.error(f"Could not init OpenAI client in hybrid path: {e}")
                return self._analyze_rule_based(df)
        
        # ---------- 1) Run cheap model over everything ----------
        nano_system_prompt = f"""Analyze each social media comment from an entertainment audience.
For each comment, output:
- emotion: one of {', '.join(self.emotion_categories)}
- score: number between -1 (very negative) and +1 (very positive)
- sarcastic: true/false
- confidence: number between 0 and 1 for how sure you are (0=guessing, 1=very sure).

IMPORTANT: You MUST output valid JSON array only. No markdown, no explanations."""

        nano_results: List[Dict] = []
        
        for i in range(0, len(comments), batch_size):
            batch_comments = comments[i:i+batch_size]
            try:
                batch_results = self._call_openai_model_batch(
                    client=client,
                    model_name=config.SENTIMENT_MODEL_CHEAP,
                    comments=batch_comments,
                    system_prompt=nano_system_prompt,
                )
                nano_results.extend(batch_results)
            except Exception as e:
                logger.warning(f"Cheap model batch {i} failed ({e}); using rule-based fallback for that slice")
                for c in batch_comments:
                    nano_results.append(self._analyze_single_comment(c))
        
        # Pad if needed
        while len(nano_results) < len(comments):
            nano_results.append(self._analyze_single_comment(""))
        
        # ---------- 2) Choose which comments to escalate ----------
        to_escalate_indices = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            res = nano_results[idx] if idx < len(nano_results) else self._analyze_single_comment("")
            score = float(res.get("sentiment_score", 0.0))
            confidence = float(res.get("model_confidence", abs(score)))
            text = str(row.get("comment_text", "") or "")
            
            # Get likes
            likes = row.get("likes", 0)
            try:
                likes = int(likes) if pd.notna(likes) else 0
            except (ValueError, TypeError):
                likes = 0
            
            # Heuristics for escalation
            ambiguous = abs(score) < config.SENTIMENT_AMBIGUITY_BAND
            low_conf = confidence < 0.6
            long_comment = len(text.split()) > config.SENTIMENT_MAX_LENGTH_FOR_NANO_ONLY
            high_impact = likes >= config.SENTIMENT_MIN_LIKES_FOR_ESCALATION
            stan_phrase_present = any(
                phrase in text.lower() for phrase in config.SENTIMENT_STAN_PHRASES
            )
            screaming = (text.isupper() and len(text) > 5) or text.count("!") >= 3
            
            if high_impact or stan_phrase_present or screaming or (ambiguous and low_conf) or long_comment:
                to_escalate_indices.append(idx)
        
        # Apply safety cap (avoid accidentally escalating 80% of comments)
        max_escalations = math.floor(len(comments) * config.SENTIMENT_MAX_ESCALATION_FRACTION)
        if len(to_escalate_indices) > max_escalations:
            logger.info(
                f"Escalation candidates {len(to_escalate_indices)} exceed cap {max_escalations}, trimming to top {max_escalations}"
            )
            # Sort by importance (likes, then confidence) and take top N
            escalation_scores = []
            for idx in to_escalate_indices:
                row = df.iloc[idx]
                likes = row.get("likes", 0)
                try:
                    likes = int(likes) if pd.notna(likes) else 0
                except (ValueError, TypeError):
                    likes = 0
                conf = nano_results[idx].get("model_confidence", 0) if idx < len(nano_results) else 0
                escalation_scores.append((idx, likes * 1000 + conf * 100))  # Weight likes heavily
            
            escalation_scores.sort(key=lambda x: x[1], reverse=True)
            to_escalate_indices = [idx for idx, _ in escalation_scores[:max_escalations]]
        
        logger.info(
            f"Hybrid sentiment: {len(comments)} total, {len(to_escalate_indices)} escalated to {config.SENTIMENT_MODEL_MAIN}"
        )
        
        if not to_escalate_indices:
            # No escalation needed; tag model and return
            for r in nano_results:
                r["model"] = config.SENTIMENT_MODEL_CHEAP
            return nano_results
        
        # ---------- 3) Run high-accuracy model on escalated subset ----------
        escalated_comments = [comments[i] for i in to_escalate_indices]
        
        main_system_prompt = f"""You are analyzing social media comments for a television entertainment brand.
For each comment, output:
- emotion: one of {', '.join(self.emotion_categories)}
- score: number between -1 and +1
- sarcastic: true/false
- confidence: 0-1

Use entertainment context (stan culture, sarcasm, emojis, caps, etc.).
IMPORTANT: Reply with valid JSON array only. No markdown."""

        main_results_flat: List[Dict] = []
        
        for i in range(0, len(escalated_comments), batch_size):
            batch_comments = escalated_comments[i:i+batch_size]
            try:
                batch_results = self._call_openai_model_batch(
                    client=client,
                    model_name=config.SENTIMENT_MODEL_MAIN,
                    comments=batch_comments,
                    system_prompt=main_system_prompt,
                )
                main_results_flat.extend(batch_results)
            except Exception as e:
                logger.error(f"High-accuracy model escalation batch failed ({e}); keeping cheap model results for that slice")
                # Fallback: keep existing nano results, do nothing
        
        # Ensure we have enough results
        while len(main_results_flat) < len(to_escalate_indices):
            main_results_flat.append(self._analyze_single_comment(""))
        
        # Map high-accuracy results back onto overall array
        for idx, result in zip(to_escalate_indices, main_results_flat):
            if idx < len(nano_results):
                nano_results[idx] = result  # Overwrite cheap model with high-accuracy model
        
        # Tag model if not already
        for i, r in enumerate(nano_results):
            if "model" not in r:
                r["model"] = config.SENTIMENT_MODEL_CHEAP
        
        return nano_results
    
    def _analyze_with_batch_api(self, df: pd.DataFrame) -> List[Dict]:
        """
        Use OpenAI Batch API for sentiment analysis (50% cheaper, asynchronous)
        
        Workflow:
        1. Create JSONL file with all requests
        2. Upload file to OpenAI
        3. Create batch job
        4. Poll for completion
        5. Download and parse results
        
        Note: This is asynchronous - results may take up to 24 hours, but costs 50% less
        """
        import tempfile
        import os
        from pathlib import Path
        
        if df.empty:
            return []
        
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.strip() == "":
            logger.warning("OPENAI_API_KEY missing; batch API mode falling back to rule-based sentiment")
            return self._analyze_rule_based(df)
        
        client = self.openai_client
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY.strip())
                self.openai_client = client
            except Exception as e:
                logger.error(f"Could not init OpenAI client for batch API: {e}")
                return self._analyze_rule_based(df)
        
        comments = df["comment_text"].astype(str).tolist()
        model_name = config.SENTIMENT_MODEL_CHEAP  # Use cheap model for batch
        
        logger.info(f"Preparing batch API request for {len(comments)} comments...")
        
        # Step 1: Create JSONL file
        jsonl_lines = []
        for i, comment in enumerate(comments):
            request = {
                "custom_id": f"comment-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing social media sentiment in entertainment contexts. Always return valid JSON."
                        },
                        {
                            "role": "user",
                            "content": f"""Analyze this social media comment from an entertainment audience.

Comment: {comment[:500]}

Output JSON:
{{"emotion": "excitement|anger|disappointment|love|disgust|surprise|fatigue|neutral", "score": -1.0 to 1.0, "sarcastic": true/false, "confidence": 0.0 to 1.0}}"""
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": config.MAX_TOKENS_SENTIMENT
                }
            }
            jsonl_lines.append(json.dumps(request))
        
        # Step 2: Upload file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write('\n'.join(jsonl_lines))
                temp_file_path = f.name
            
            logger.info(f"Uploading batch input file ({len(jsonl_lines)} requests)...")
            with open(temp_file_path, 'rb') as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            os.unlink(temp_file_path)  # Clean up temp file
            logger.info(f"File uploaded: {uploaded_file.id}")
            
            # Step 3: Create batch job
            logger.info("Creating batch job...")
            batch = client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            logger.info(f"Batch job created: {batch.id}")
            logger.info(f"Status: {batch.status}")
            logger.info("Batch API jobs are processed asynchronously (up to 24 hours).")
            logger.info("Results will be available when batch completes. Polling for completion...")
            
            # Step 4: Poll for completion
            start_time = time.time()
            while batch.status in ["validating", "in_progress"]:
                if time.time() - start_time > config.SENTIMENT_BATCH_MAX_WAIT:
                    logger.warning(f"Batch {batch.id} exceeded max wait time. Returning partial results.")
                    break
                
                time.sleep(config.SENTIMENT_BATCH_POLL_INTERVAL)
                batch = client.batches.retrieve(batch.id)
                logger.info(f"Batch {batch.id} status: {batch.status}")
            
            if batch.status != "completed":
                logger.error(f"Batch {batch.id} failed with status: {batch.status}")
                logger.info("Falling back to rule-based analysis for all comments")
                return self._analyze_rule_based(df)
            
            # Step 5: Download results
            logger.info(f"Batch completed! Downloading results from {batch.output_file_id}...")
            output_file = client.files.content(batch.output_file_id)
            
            # Parse results
            results = []
            result_map = {}
            
            # Parse JSONL output
            for line in output_file.text.split('\n'):
                if not line.strip():
                    continue
                try:
                    result_data = json.loads(line)
                    custom_id = result_data.get("custom_id", "")
                    if result_data.get("response", {}).get("status_code") == 200:
                        response_body = result_data["response"]["body"]
                        if isinstance(response_body, str):
                            response_body = json.loads(response_body)
                        
                        choice = response_body.get("choices", [{}])[0]
                        message_content = choice.get("message", {}).get("content", "{}")
                        
                        if isinstance(message_content, str):
                            try:
                                sentiment_data = json.loads(message_content)
                            except:
                                # Fallback parsing
                                sentiment_data = self._parse_json_with_repair(message_content)
                                if isinstance(sentiment_data, list) and sentiment_data:
                                    sentiment_data = sentiment_data[0]
                        else:
                            sentiment_data = message_content
                        
                        result_map[custom_id] = {
                            "primary_emotion": sentiment_data.get("emotion", "neutral"),
                            "sentiment_score": float(sentiment_data.get("score", 0.0)),
                            "is_sarcastic": bool(sentiment_data.get("sarcastic", False)),
                            "secondary_emotions": [],
                            "model": model_name,
                            "model_confidence": float(sentiment_data.get("confidence", 0.5)),
                        }
                    else:
                        # Request failed, use fallback
                        logger.warning(f"Request {custom_id} failed in batch, using rule-based fallback")
                        comment_idx = int(custom_id.split('-')[1]) if '-' in custom_id else 0
                        if comment_idx < len(comments):
                            result_map[custom_id] = self._analyze_single_comment(comments[comment_idx])
                except Exception as e:
                    logger.warning(f"Error parsing batch result line: {e}")
                    continue
            
            # Map results back to original order
            for i in range(len(comments)):
                custom_id = f"comment-{i}"
                if custom_id in result_map:
                    results.append(result_map[custom_id])
                else:
                    # Fallback for missing results
                    results.append(self._analyze_single_comment(comments[i]))
            
            # Cost tracking (Batch API is 50% of regular pricing)
            try:
                # Estimate from batch metadata if available
                if hasattr(batch, 'request_counts'):
                    total_requests = batch.request_counts.get('total', len(comments))
                    # Rough estimate: assume average tokens per request
                    # Batch API pricing is 50% of regular
                    estimated_input_tokens = total_requests * 100  # Rough estimate
                    estimated_output_tokens = total_requests * 50
                    
                    if "gpt-5-nano" in model_name.lower():
                        input_cost = (estimated_input_tokens / 1_000_000 * 0.05) * 0.5  # 50% discount
                        output_cost = (estimated_output_tokens / 1_000_000 * 0.40) * 0.5
                    else:
                        input_cost = (estimated_input_tokens / 1_000_000 * 0.15) * 0.5
                        output_cost = (estimated_output_tokens / 1_000_000 * 0.60) * 0.5
                    
                    self.estimated_cost += input_cost + output_cost
            except Exception as e:
                logger.debug(f"Could not estimate batch API cost: {e}")
            
            logger.info(f"Batch API processing complete. Processed {len(results)} comments.")
            return results
            
        except Exception as e:
            logger.error(f"Batch API processing failed: {e}. Falling back to rule-based analysis.")
            import traceback
            logger.debug(traceback.format_exc())
            return self._analyze_rule_based(df)



