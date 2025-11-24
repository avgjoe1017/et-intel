"""
Intelligence Brief Report Generator
Creates professional PDF reports with executive summary and visualizations
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import logging
from .. import config

logger = logging.getLogger(__name__)

class IntelligenceBriefGenerator:
    """
    Generates professional PDF Intelligence Briefs
    """
    
    def __init__(self):
        self.reports_dir = config.REPORTS_DIR
        self.charts_dir = config.REPORTS_DIR / "charts"
        self.charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def generate_report(self, brief_data: dict, output_filename: str = None) -> Path:
        """
        Generate PDF report from intelligence brief data
        
        Args:
            brief_data: Dict from pipeline.generate_intelligence_brief()
            output_filename: Optional custom filename
        
        Returns:
            Path to generated PDF
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ET_Intelligence_Brief_{timestamp}.pdf"
        
        output_path = self.reports_dir / output_filename
        
        # Add story summary to brief data (computed here for report)
        if config.USE_LLM_ENHANCEMENT and config.OPENAI_API_KEY:
            try:
                brief_data['story_summary'] = self.extract_story_with_gpt(brief_data)
                brief_data['recommendations'] = self.generate_recommendations_with_gpt(brief_data)
            except Exception as e:
                logger.warning(f"LLM story/recommendations failed, using rule-based: {e}")
                brief_data['story_summary'] = self.extract_story_summary(brief_data)
                brief_data['recommendations'] = self.generate_recommendations(brief_data)
        else:
            brief_data['story_summary'] = self.extract_story_summary(brief_data)
            brief_data['recommendations'] = self.generate_recommendations(brief_data)
        
        # Create PDF
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(brief_data))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(brief_data))
        story.append(PageBreak())
        
        # Velocity alerts (if any)
        if brief_data.get('velocity_alerts'):
            story.extend(self._create_velocity_section(brief_data))
            story.append(PageBreak())
        
        # Top entities section
        story.extend(self._create_entities_section(brief_data))
        story.append(PageBreak())
        
        # Storylines (moved up - more important than raw entity counts)
        story.extend(self._create_storylines_section(brief_data))
        story.append(PageBreak())
        
        # Sentiment analysis
        story.extend(self._create_sentiment_section(brief_data))
        story.append(PageBreak())
        
        # High Priority Alerts (replaces Recommendations - just flags, no prescriptions)
        recommendations = brief_data.get('recommendations', [])
        if recommendations:
            story.append(PageBreak())
            story.append(Paragraph("HIGH PRIORITY ALERTS", getSampleStyleSheet()['Heading1']))
            story.append(Spacer(1, 0.2*inch))
            
            for rec in recommendations[:5]:  # Top 5 alerts
                # Priority and Entity header (heat map style)
                priority_colors = {
                    'HIGH': colors.red,
                    'MEDIUM': colors.orange,
                    'LOW': colors.grey
                }
                header_style = ParagraphStyle(
                    'AlertHeader',
                    parent=getSampleStyleSheet()['Heading3'],
                    textColor=priority_colors.get(rec.get('priority', 'MEDIUM'), colors.black)
                )
                entity = rec.get('entity', 'Unknown')
                priority = rec.get('priority', 'MEDIUM')
                story.append(Paragraph(f"{priority}: {entity}", header_style))
                
                # Just show the reason (facts only - no prescriptions)
                body_style = getSampleStyleSheet()['Normal']
                reason = rec.get('reason', '')
                # Remove any prescriptive language
                reason = reason.replace('ET should', '').replace('ET must', '')
                reason = reason.replace('Recommendation:', '').replace('Action:', '')
                reason = reason.replace('Suggestion:', '').strip()
                story.append(Paragraph(reason, body_style))
                story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def extract_story_summary(self, brief_data):
        """
        Extract the actual story from the data
        Use top comments, sentiment, and storylines to build narrative
        """
        # Get top entities
        top_entities = sorted(
            [(k, v) for k, v in brief_data['sentiment_summary'].items()],
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:3]
        
        summary = []
        
        for entity_name, entity_data in top_entities:
            if entity_data['total_mentions'] < 10:
                continue
                
            # Build story for this entity
            sentiment = entity_data['avg_sentiment']
            weighted = entity_data['weighted_avg_sentiment']
            top_emotion = max(entity_data['emotion_breakdown'].items(), key=lambda x: x[1])[0] if entity_data['emotion_breakdown'] else 'neutral'
            top_comment = entity_data.get('top_liked_comment', {})
            
            # Sentiment characterization
            if sentiment < -0.3:
                sentiment_word = "strongly negative"
            elif sentiment < -0.1:
                sentiment_word = "negative"
            elif sentiment < 0.1:
                sentiment_word = "mixed"
            elif sentiment < 0.3:
                sentiment_word = "positive"
            else:
                sentiment_word = "strongly positive"
            
            # Build narrative
            story = f"<b>{entity_name}</b>: {sentiment_word} ({sentiment:.2f})"
            
            # Add weighted insight
            if weighted - sentiment > 0.2:
                story += f" - but positive comments are resonating ({weighted:.2f} weighted)"
            elif weighted - sentiment < -0.2:
                story += f" - and negative comments are amplified by likes ({weighted:.2f} weighted)"
            
            # Add emotion
            story += f". Primary emotion: {top_emotion}"
            
            # Add top comment context
            if top_comment and top_comment.get('text'):
                text = top_comment['text'].replace('\n', ' ')
                story += f". Top comment ({top_comment['likes']:,} likes): \"{text[:100]}...\""
            
            summary.append(story)
        
        return "<br/><br/>".join(summary)
    
    def extract_story_with_gpt(self, brief_data):
        """
        Use GPT-4o-mini to synthesize what's actually happening from the data
        Cost: ~$0.01 per brief
        """
        if not config.OPENAI_API_KEY or not config.OPENAI_API_KEY.strip():
            logger.warning("OpenAI API key not available for LLM story extraction")
            return self.extract_story_summary(brief_data)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client for story extraction: {e}")
            return self.extract_story_summary(brief_data)
        
        # Get top 5 comments by likes
        top_comments = []
        for entity_name, entity_data in brief_data.get('sentiment_summary', {}).items():
            top_comment = entity_data.get('top_liked_comment')
            if top_comment and top_comment.get('likes', 0) > 10:
                top_comments.append({
                    'entity': entity_name,
                    'text': top_comment.get('text', '')[:200],
                    'likes': top_comment.get('likes', 0),
                    'sentiment': top_comment.get('sentiment', 0)
                })
        
        top_comments = sorted(top_comments, key=lambda x: x['likes'], reverse=True)[:5]
        
        # Get top entities sentiment data
        sentiment_data = {}
        for entity_name, entity_data in list(brief_data.get('sentiment_summary', {}).items())[:5]:
            sentiment_data[entity_name] = {
                'sentiment': entity_data.get('avg_sentiment', 0),
                'weighted_sentiment': entity_data.get('weighted_avg_sentiment', 0),
                'mentions': entity_data.get('total_mentions', 0),
                'top_emotion': max(entity_data.get('emotion_breakdown', {}).items(), key=lambda x: x[1])[0] if entity_data.get('emotion_breakdown') else 'neutral'
            }
        
        prompt = f"""Analyze these Instagram comments and extract the story:

Top 5 comments (by likes):
{json.dumps(top_comments, indent=2)}

Sentiment data for top entities:
{json.dumps(sentiment_data, indent=2)}

Storylines detected:
{json.dumps(brief_data.get('top_storylines', [])[:3], indent=2)}

Answer in 3-4 paragraphs:
1. What is the dominant narrative?
2. Why is sentiment positive/negative?
3. What are people actually discussing?
4. What patterns emerge from high-engagement comments?

Be specific and cite comment examples. Write in a professional, analytical tone suitable for an entertainment news intelligence brief."""

        try:
            response = client.chat.completions.create(
                model=config.SENTIMENT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert social media intelligence analyst specializing in entertainment news sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=config.MAX_TOKENS_STORY
            )
            
            story_text = response.choices[0].message.content
            
            # Convert to HTML format for reportlab
            story_html = story_text.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
            
            logger.info("LLM story extraction completed successfully")
            return story_html
            
        except Exception as e:
            logger.warning(f"LLM story extraction failed: {e}. Using rule-based fallback.")
            return self.extract_story_summary(brief_data)
    
    def extract_storylines_with_context(self, brief_data):
        """
        Turn keyword counts into actual storylines with context using LLM
        Cost: ~$0.01 per brief
        """
        if not config.USE_LLM_ENHANCEMENT or not config.OPENAI_API_KEY or not config.OPENAI_API_KEY.strip():
            return None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client for storyline extraction: {e}")
            return None
        
        # Get raw storylines (keyword counts)
        raw_storylines = brief_data.get('top_storylines', [])[:5]
        if not raw_storylines:
            return None
        
        # Get top entities with sentiment
        top_entities = []
        for entity_name, entity_data in list(brief_data.get('sentiment_summary', {}).items())[:5]:
            emotion_breakdown = entity_data.get('emotion_breakdown', {})
            top_emotion = max(emotion_breakdown.items(), key=lambda x: x[1])[0] if emotion_breakdown else 'neutral'
            top_entities.append({
                'entity': entity_name,
                'sentiment': entity_data.get('avg_sentiment', 0),
                'weighted_sentiment': entity_data.get('weighted_avg_sentiment', 0),
                'mentions': entity_data.get('total_mentions', 0),
                'likes': entity_data.get('total_likes', 0),
                'top_emotion': top_emotion,
                'top_comment': entity_data.get('top_liked_comment', {})
            })
        
        # Get top comments across all entities (truncate and clean for JSON safety)
        all_top_comments = []
        for entity_name, entity_data in brief_data.get('sentiment_summary', {}).items():
            top_comment = entity_data.get('top_liked_comment')
            if top_comment and top_comment.get('likes', 0) > 10:
                # Clean comment text: remove newlines, truncate, escape quotes
                comment_text = str(top_comment.get('text', ''))
                comment_text = comment_text.replace('\n', ' ').replace('\r', ' ')
                comment_text = comment_text[:200]  # Shorter to avoid JSON issues
                all_top_comments.append({
                    'entity': entity_name,
                    'text': comment_text,
                    'likes': top_comment.get('likes', 0),
                    'sentiment': top_comment.get('sentiment', 0)
                })
        all_top_comments = sorted(all_top_comments, key=lambda x: x['likes'], reverse=True)[:10]
        
        # Format comments as numbered list (don't include in JSON to avoid parsing issues)
        comments_list = "\n".join([
            f"[{i+1}] {c['entity']}: {c['text'][:150]}... ({c['likes']} likes, sentiment {c['sentiment']:.2f})"
            for i, c in enumerate(all_top_comments)
        ])
        
        prompt = f"""Based on this social media data, identify the top 3-4 story beats. Just report what's happening - facts only, no recommendations.

Storyline keywords detected:
{json.dumps(raw_storylines, indent=2)}

Top entities and their sentiment:
{json.dumps(top_entities, indent=2)}

Top 10 most-liked comments (reference by number [1], [2], etc.):
{comments_list}

For each story beat, provide:
1. **Urgency indicator**: HIGH ACTIVITY / EMERGING / DEVELOPING
2. **Headline**: 5-7 word specific story title with names
3. **What's happening**: 2-3 sentences stating facts - who, what, sentiment, emotion breakdown, top evidence
4. **Evidence citation**: Quote the specific comment text that supports this beat (use evidence_comment_index)

CRITICAL RULES:
- DO NOT suggest actions. DO NOT tell ET what to do. Just report the facts.
- Be SPECIFIC - use names, numbers, and quote evidence.
- Understand CONTEXT: If someone has negative sentiment (-0.6) but comments DEFEND them (e.g., "he's honest", "I liked Justin"), the story beat should say they're "defended" or "praised", not "discussed negatively"
- If sentiment is negative but comments are supportive, explain the context (e.g., "Despite negative context, audience defends X")
- Use evidence_comment_index (1-10) to reference which comment supports this beat.
- Keep all text fields simple - avoid quotes, newlines, or special characters.
- ONLY create story beats for which you can cite specific evidence from the comments above. Do NOT hallucinate or infer stories not present in the data.

Return JSON in this exact format:
{{
  "story_beats": [
    {{
      "urgency": "HIGH ACTIVITY",
      "headline": "Blake Lively Casting Rejection",
      "description": "2-3 sentences stating facts - who, what, sentiment, emotion breakdown, top evidence",
      "mention_count": 114,
      "percentage": 18.7,
      "sentiment": -0.46,
      "key_emotion": "anger",
      "emotion_breakdown": "36% anger, 23% disappointment",
      "evidence_comment_index": 1,
      "evidence_likes": 4890
    }}
  ]
}}"""

        try:
            response = client.chat.completions.create(
                model=config.SENTIMENT_MODEL_MAIN if hasattr(config, 'SENTIMENT_MODEL_MAIN') else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert entertainment news analyst. You transform keyword counts into detailed story beats with context. You MUST return valid JSON only. IMPORTANT: Understand context - if someone has negative sentiment but comments DEFEND them (e.g., 'he's honest', 'I liked Justin'), they are being defended/praised, not criticized. Only create story beats for which you can cite specific evidence from the comments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=config.MAX_TOKENS_STORY if hasattr(config, 'MAX_TOKENS_STORY') else 2000,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            
            # Try to parse JSON with repair attempts
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                # Try to repair common JSON issues
                # Remove markdown code blocks if present
                if '```json' in raw_content:
                    raw_content = raw_content.split('```json')[1].split('```')[0].strip()
                elif '```' in raw_content:
                    raw_content = raw_content.split('```')[1].split('```')[0].strip()
                
                # Try to fix common issues
                import re
                # Remove trailing commas
                raw_content = re.sub(r',\s*}', '}', raw_content)
                raw_content = re.sub(r',\s*]', ']', raw_content)
                
                # Try to fix unterminated strings by finding and closing them
                # This is a simple heuristic - if we find an unterminated string, try to close it
                lines = raw_content.split('\n')
                repaired_lines = []
                for line in lines:
                    # Count quotes - if odd number, might be unterminated
                    quote_count = line.count('"')
                    if quote_count % 2 == 1 and '"' in line:
                        # Try to close the string at the end of the line
                        if not line.rstrip().endswith('"') and not line.rstrip().endswith('\\'):
                            # Find last unescaped quote and see if we need to close
                            # Simple fix: if line ends with text after quote, add closing quote
                            if ':' in line and line.split(':')[-1].strip().startswith('"'):
                                # Might be unterminated value
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    key_part = parts[0]
                                    value_part = parts[1].strip()
                                    if value_part.startswith('"') and not value_part.endswith('"'):
                                        # Try to close it
                                        if not value_part.endswith('",') and not value_part.endswith('"'):
                                            line = key_part + ': ' + value_part + '",'
                    repaired_lines.append(line)
                raw_content = '\n'.join(repaired_lines)
                
                # Try parsing again
                try:
                    result = json.loads(raw_content)
                except json.JSONDecodeError as e2:
                    # Last resort: try regex-based extraction for critical fields
                    logger.warning(f"JSON parse failed, attempting regex extraction: {e2}")
                    try:
                        import re
                        # Extract story beats using regex patterns (handle both old and new format)
                        storylines = []
                        # Try new format first (story_beats with urgency, headline)
                        storybeat_pattern = r'\{\s*"urgency"\s*:\s*"([^"]+)"[^}]*"headline"\s*:\s*"([^"]+)"[^}]*"description"\s*:\s*"([^"]+)"'
                        matches = re.finditer(storybeat_pattern, raw_content, re.DOTALL)
                        for match in matches:
                            storylines.append({
                                'urgency': match.group(1),
                                'headline': match.group(2),
                                'description': match.group(3)
                            })
                        
                        # If no matches, try old format (title, storyline, description)
                        if not storylines:
                            storyline_pattern = r'\{\s*"title"\s*:\s*"([^"]+)"[^}]*"description"\s*:\s*"([^"]+)"'
                            matches = re.finditer(storyline_pattern, raw_content, re.DOTALL)
                            for match in matches:
                                storylines.append({
                                    'headline': match.group(1),
                                    'title': match.group(1),
                                    'description': match.group(2)
                                })
                        
                        if storylines:
                            logger.info(f"Regex extraction succeeded: found {len(storylines)} storylines")
                            result = {'storylines': storylines}
                        else:
                            logger.warning(f"Regex extraction found no storylines. Raw response (first 500 chars): {raw_content[:500]}")
                            return None
                    except Exception as e3:
                        logger.warning(f"Regex extraction also failed: {e3}")
                        logger.debug(f"Raw response (first 500 chars): {raw_content[:500]}")
                        return None
            
            # Handle both old format (storylines) and new format (story_beats)
            storylines = result.get('story_beats', result.get('storylines', []))
            
            # Validate and normalize storylines structure
            valid_storylines = []
            for sl in storylines:
                # Accept either 'title' or 'headline' for the title field
                has_title = isinstance(sl, dict) and ('title' in sl or 'headline' in sl) and 'description' in sl
                if has_title:
                    # Normalize evidence format (handle multiple formats)
                    if 'evidence_comment_index' in sl:
                        # New format: reference by index
                        comment_idx = sl['evidence_comment_index'] - 1  # Convert to 0-based
                        if 0 <= comment_idx < len(all_top_comments):
                            sl['evidence_text'] = all_top_comments[comment_idx]['text'][:150]
                            sl['evidence_likes'] = all_top_comments[comment_idx]['likes']
                        else:
                            sl['evidence_text'] = ''
                            sl['evidence_likes'] = 0
                    elif 'evidence' in sl and isinstance(sl['evidence'], dict):
                        # Old format: evidence is a dict
                        sl['evidence_text'] = sl['evidence'].get('text', '')
                        sl['evidence_likes'] = sl['evidence'].get('likes', 0)
                    elif 'evidence_text' in sl:
                        # Already has evidence_text
                        sl['evidence_likes'] = sl.get('evidence_likes', 0)
                    else:
                        # No evidence provided
                        sl['evidence_text'] = ''
                        sl['evidence_likes'] = 0
                    
                    valid_storylines.append(sl)
                else:
                    logger.warning(f"Skipping invalid storyline entry: {sl}")
            
            if valid_storylines:
                logger.info(f"LLM storyline extraction: {len(raw_storylines)} keywords -> {len(valid_storylines)} detailed storylines")
                return valid_storylines
            else:
                logger.warning("LLM storyline extraction returned no valid storylines")
                return None
                
        except Exception as e:
            logger.warning(f"LLM storyline extraction failed: {e}. Using rule-based fallback.")
            return None

    def generate_recommendations(self, brief_data):
        """
        Generate actionable recommendations based on sentiment and storylines
        """
        recommendations = []
        
        # Analyze top entities
        for entity_name, entity_data in brief_data['sentiment_summary'].items():
            if entity_data['total_mentions'] < 20:
                continue
            
            sentiment = entity_data['avg_sentiment']
            weighted = entity_data['weighted_avg_sentiment']
            
            # Get storylines safely
            storylines = []
            if 'top_storylines' in brief_data and brief_data['top_storylines']:
                storylines = [s['storyline'] for s in brief_data['top_storylines']]
            
            # Negative sentiment + high engagement
            if sentiment < -0.3 and entity_data['total_likes'] > 1000:
                recommendations.append({
                    'entity': entity_name,
                    'priority': 'HIGH',
                    'action': 'INVESTIGATE',
                    'reason': f"Strong negative sentiment ({sentiment:.2f}) with high engagement ({entity_data['total_likes']:,} likes). Potential controversy brewing.",
                    'suggestion': f"ET should investigate what's driving negative sentiment and consider covering the controversy angle."
                })
            
            # Positive sentiment + trending
            elif sentiment > 0.3 and entity_data['total_mentions'] > 50:
                recommendations.append({
                    'entity': entity_name,
                    'priority': 'MEDIUM',
                    'action': 'PROMOTE',
                    'reason': f"Positive sentiment ({sentiment:.2f}) with strong engagement.",
                    'suggestion': f"ET should increase coverage - audience is receptive."
                })
            
            # Controversy storyline active
            if 'controversy' in storylines or 'lawsuit' in storylines:
                recommendations.append({
                    'entity': entity_name,
                    'priority': 'HIGH',
                    'action': 'MONITOR',
                    'reason': "Controversy/lawsuit storyline detected",
                    'suggestion': "Track sentiment velocity closely. Prepare coverage if story escalates."
                })
        
        return recommendations
    
    def generate_recommendations_with_gpt(self, brief_data):
        """
        Use GPT-4o-mini to generate actionable recommendations
        Cost: ~$0.01 per brief
        """
        if not config.OPENAI_API_KEY or not config.OPENAI_API_KEY.strip():
            logger.warning("OpenAI API key not available for LLM recommendations")
            return self.generate_recommendations(brief_data)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client for recommendations: {e}")
            return self.generate_recommendations(brief_data)
        
        # Prepare story summary
        story_summary = brief_data.get('story_summary', 'No story summary available')
        
        # Prepare sentiment trends
        trends = brief_data.get('trends', {})
        trends_text = "No trend data available"
        if trends.get('trends'):
            trends_list = trends['trends'][:5]
            trends_text = json.dumps(trends_list, indent=2)
        
        # Get top entities with sentiment
        top_entities = []
        for entity_name, entity_data in list(brief_data.get('sentiment_summary', {}).items())[:5]:
            top_entities.append({
                'entity': entity_name,
                'sentiment': entity_data.get('avg_sentiment', 0),
                'weighted_sentiment': entity_data.get('weighted_avg_sentiment', 0),
                'mentions': entity_data.get('total_mentions', 0),
                'likes': entity_data.get('total_likes', 0)
            })
        
        prompt = f"""Based on this social media intelligence brief, what should Entertainment Tonight's editorial team do?

Story Summary:
{story_summary}

Sentiment Trends:
{trends_text}

Top Entities:
{json.dumps(top_entities, indent=2)}

Storylines:
{json.dumps(brief_data.get('top_storylines', [])[:3], indent=2)}

Provide 3-5 specific, actionable recommendations. For each recommendation, return JSON format:
{{
  "recommendations": [
    {{
      "priority": "HIGH|MEDIUM|LOW",
      "entity": "Entity Name",
      "action": "INVESTIGATE|PROMOTE|MONITOR|AVOID",
      "reason": "2-3 sentence explanation",
      "suggestion": "Specific actionable suggestion for ET"
    }}
  ]
}}

Focus on what ET should actually DO with this intelligence."""

        try:
            response = client.chat.completions.create(
                model=config.SENTIMENT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert media strategy consultant specializing in entertainment news editorial decisions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=config.MAX_TOKENS_RECOMMENDATIONS
            )
            
            response_text = response.choices[0].message.content
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            recommendations = result.get('recommendations', [])
            
            # Validate recommendations format
            validated = []
            for rec in recommendations:
                if all(key in rec for key in ['priority', 'entity', 'action', 'reason', 'suggestion']):
                    validated.append(rec)
            
            if validated:
                logger.info(f"LLM generated {len(validated)} recommendations")
                return validated
            else:
                logger.warning("LLM recommendations format invalid, using rule-based fallback")
                return self.generate_recommendations(brief_data)
            
        except Exception as e:
            logger.warning(f"LLM recommendations generation failed: {e}. Using rule-based fallback.")
            return self.generate_recommendations(brief_data)

    def interpret_weighted_sentiment(self, raw, weighted):
        """
        Correctly interpret the delta between raw and weighted sentiment
        
        weighted > raw: Top comments (by likes) are LESS negative than average
        weighted < raw: Top comments (by likes) are MORE negative than average
        """
        if raw is None or weighted is None:
            return "Insufficient data"
            
        delta = weighted - raw
        abs_delta = abs(delta)
        
        if abs_delta < 0.1:
            return f"Balanced agreement (Δ{delta:+.2f}) - top comments align with average sentiment"
        
        if delta > 0:
            # Weighted is MORE positive (less negative) than raw
            # This means highly-liked comments are pulling the average UP (less negative)
            if raw < 0:
                # Both negative, but weighted less negative
                return f"Top comments less negative than average (Δ{delta:+.2f}) - highly-liked comments pulling sentiment higher"
            else:
                # Raw positive, weighted even more positive
                return f"Top comments more positive than average (Δ{delta:+.2f}) - highly-liked comments amplifying positive sentiment"
        else:
            # Weighted is MORE negative than raw
            # This means highly-liked comments are pulling the average DOWN (more negative)
            if raw > 0:
                # Raw positive, but weighted less positive (or negative)
                return f"Top comments less positive than average (Δ{delta:+.2f}) - highly-liked comments pulling sentiment lower"
            else:
                # Both negative, but weighted more negative
                return f"Top comments more negative than average (Δ{delta:+.2f}) - highly-liked comments amplifying negative sentiment"

    def _create_title_page(self, data: dict) -> list:
        """Create title page"""
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=32,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        story = []
        
        # Add spacing
        story.append(Spacer(1, 2*inch))
        
        # Title
        story.append(Paragraph(config.REPORT_TITLE, title_style))
        story.append(Paragraph(config.REPORT_SUBTITLE, subtitle_style))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Date range (Priority 5 Fix)
        date_range = data['metadata']['date_range']
        start = datetime.fromisoformat(date_range['start'])
        end = datetime.fromisoformat(date_range['end'])
        
        # Don't show date range if it includes 1970 (bad timestamps)
        if start.year == 1970:
            date_range_text = f"Through {end.strftime('%B %d, %Y')}"
        else:
            date_range_text = f"{start.strftime('%B %d, %Y')} - {end.strftime('%B %d, %Y')}"
        
        story.append(Paragraph(f"<b>Period:</b> {date_range_text}", subtitle_style))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
            subtitle_style
        ))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Key metrics box
        metrics_data = [
            ['Total Comments Analyzed', f"{data['metadata']['total_comments']:,}"],
            ['Unique Posts', f"{data['metadata']['unique_posts']:,}"],
            ['Platforms', ', '.join(data['metadata']['platforms'])],
            ['Entities Tracked', f"{len(data['entities']['people']) + len(data['entities']['shows'])}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.white)
        ]))
        
        story.append(metrics_table)
        
        return story
    
    def _create_executive_summary(self, data: dict) -> list:
        """Create executive summary section"""
        styles = getSampleStyleSheet()
        story = []
        
        # Section header
        story.append(Paragraph("EXECUTIVE SUMMARY", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Get top 5 entities by sentiment
        sentiment_summary = data['sentiment_summary']
        
        # Sort by mention count
        sorted_entities = sorted(
            sentiment_summary.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:5]
        
        # Key findings - structured format
        story.append(Paragraph("<b>KEY FINDINGS:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        findings = []
        
        # Volume section
        if sorted_entities:
            volume_items = []
            for entity_name, entity_data in sorted_entities[:3]:  # Top 3 entities
                total = entity_data.get('total_mentions', 0)
                total_comments = len(data.get('comments', [])) if 'comments' in data else 609  # Fallback
                percentage = (total / total_comments * 100) if total_comments > 0 else 0
                volume_items.append(f"<b>{entity_name}</b>: {total:,} mentions ({percentage:.0f}% of tracked entities)")
            
            if volume_items:
                findings.append(("<b>📊 VOLUME</b>", "header"))
                for item in volume_items:
                    findings.append((item, "item"))
        
        # Sentiment section
        if sorted_entities:
            top_entity, top_data = sorted_entities[0]
            sentiment = top_data.get('avg_sentiment', 0)
            weighted = top_data.get('weighted_avg_sentiment', 0)
            total_likes = top_data.get('total_likes', 0)
            top_comment = top_data.get('top_liked_comment', {})
            
            sentiment_items = []
            sentiment_items.append(
                f"<b>{top_entity}</b>: {sentiment:.2f} raw, {weighted:.2f} weighted"
            )
            if weighted < sentiment:
                sentiment_items.append("(highly-liked comments amplifying negativity)")
            elif weighted > sentiment:
                sentiment_items.append("(highly-liked comments pulling sentiment higher)")
            
            if top_comment and top_comment.get('likes', 0) > 0:
                comment_text = top_comment.get('text', '')[:80].replace('\n', ' ')
                if len(top_comment.get('text', '')) > 80:
                    comment_text += "..."
                sentiment_items.append(
                    f"Top comment ({top_comment['likes']:,} likes): \"{comment_text}\""
                )
            
            if sentiment_items:
                findings.append(("<b>📉 SENTIMENT</b>", "header"))
                for item in sentiment_items:
                    findings.append((item, "item"))
        
        # Alerts section
        alert_items = []
        
        # Check for controversy/lawsuit storylines
        if data.get('top_storylines'):
            controversy_storylines = [s for s in data['top_storylines'] if s.get('storyline', '').lower() in ['controversy', 'lawsuit']]
            if controversy_storylines:
                total_controversy_mentions = sum(s.get('mention_count', 0) for s in controversy_storylines)
                affected_entities = []
                for entity_name, entity_data in sorted_entities[:3]:
                    if entity_data.get('total_mentions', 0) > 0:
                        affected_entities.append(entity_name)
                if affected_entities:
                    alert_items.append(
                        f"Controversy/lawsuit storyline detected across {len(affected_entities)} entities ({total_controversy_mentions} mentions)"
                    )
        
        # Check emotion breakdown
        if sorted_entities:
            top_entity, top_data = sorted_entities[0]
            emotion_breakdown = top_data.get('emotion_breakdown', {})
            if emotion_breakdown:
                total_emotions = sum(emotion_breakdown.values())
                if total_emotions > 0:
                    top_emotions = sorted(emotion_breakdown.items(), key=lambda x: x[1], reverse=True)[:2]
                    if top_emotions:
                        emotion_text = ", ".join([f"{round(count / total_emotions * 100)}% {emotion}" for emotion, count in top_emotions])
                        alert_items.append(f"Strong negative engagement: {emotion_text}")
        
        if alert_items:
            findings.append(("<b>🚨 ALERTS</b>", "header"))
            for item in alert_items:
                findings.append((item, "item"))
        
        # Display findings
        for finding in findings:
            if isinstance(finding, tuple):
                finding_text, finding_type = finding
                if finding_type == "header":
                    story.append(Paragraph(finding_text, styles['Heading3']))
                    story.append(Spacer(1, 0.05*inch))
                else:
                    story.append(Paragraph(f"• {finding_text}", styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            else:
                story.append(Paragraph(f"• {finding}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
        # Narrative Summary (Priority 3)
        if data.get('story_summary'):
            story.append(Paragraph("<b>NARRATIVE ANALYSIS:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(data['story_summary'], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_velocity_section(self, data: dict) -> list:
        """Create velocity alerts section"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("VELOCITY ALERTS - RISK RADAR", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(
            "The following entities experienced significant sentiment velocity changes (±30% or more) "
            "within the past 72 hours:",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Create table
        table_data = [['Entity', 'Previous', 'Recent', 'Change', 'Status']]
        
        for alert in data['velocity_alerts'][:10]:  # Top 10 alerts
            status = "⚠ FALLING" if alert['sentiment_change'] < 0 else "↑ RISING"
            
            table_data.append([
                alert['entity'],
                f"{alert['previous_sentiment']:.2f}",
                f"{alert['recent_sentiment']:.2f}",
                f"{alert['percent_change']:+.1f}%",
                status
            ])
        
        t = Table(table_data, colWidths=[2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(t)
        
        return story
    
    def _create_entities_section(self, data: dict) -> list:
        """Create top entities section with chart"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("TOP ENTITIES BY VOLUME", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Create chart
        chart_path = self._generate_entity_chart(data)
        if chart_path and chart_path.exists():
            img = RLImage(str(chart_path), width=6*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        # Top 10 table
        sentiment_summary = data['sentiment_summary']
        sorted_entities = sorted(
            sentiment_summary.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:10]
        
        table_data = [['Rank', 'Entity', 'Mentions', 'Likes', 'Raw Sent', 'Weighted Sent', 'Delta']]
        
        for i, (entity, entity_data) in enumerate(sorted_entities, 1):
            total = entity_data.get('total_mentions', 0)
            total_likes = entity_data.get('total_likes', 0)
            raw_sent = entity_data.get('avg_sentiment', 0)
            weighted_sent = entity_data.get('weighted_avg_sentiment', raw_sent)
            delta = weighted_sent - raw_sent
            
            # Color code delta (positive = agreement with sentiment, negative = disagreement)
            delta_str = f"{delta:+.2f}"
            if abs(delta) > 0.1:
                delta_str = f"{delta:+.2f} [WARN]" if delta < 0 else f"{delta:+.2f} [OK]"
            
            table_data.append([
                str(i),
                entity[:20],
                f"{total:,}",
                f"{total_likes:,}",
                f"{raw_sent:+.2f}",
                f"{weighted_sent:+.2f}",
                delta_str
            ])
        
        t = Table(table_data, colWidths=[0.5*inch, 1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.7*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Add detailed weighted sentiment analysis for top 3 entities
        story.append(Paragraph("WEIGHTED SENTIMENT ANALYSIS (Like-Weighted)", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Weighted sentiment accounts for comment likes (agreement). A comment with 1000 likes represents "
            "1000+ people's opinions, not just one. This reveals what the community actually agrees with.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        for i, (entity, entity_data) in enumerate(sorted_entities[:3], 1):
            raw_sent = entity_data.get('avg_sentiment', 0)
            weighted_sent = entity_data.get('weighted_avg_sentiment', raw_sent)
            total_likes = entity_data.get('total_likes', 0)
            delta = weighted_sent - raw_sent
            top_comment = entity_data.get('top_liked_comment')
            
            # Interpretation (Priority 2)
            interpretation = self.interpret_weighted_sentiment(raw_sent, weighted_sent)
            
            story.append(Paragraph(f"<b>{entity}</b>", styles['Heading3']))
            story.append(Paragraph(
                f"Raw sentiment: {raw_sent:+.2f} | Weighted sentiment: {weighted_sent:+.2f} | "
                f"Delta: {delta:+.2f} | Total likes: {total_likes:,}",
                styles['Normal']
            ))
            story.append(Paragraph(f"<i>{interpretation}</i>", styles['Normal']))
            
            if top_comment and top_comment.get('likes', 0) > 10:
                story.append(Paragraph(
                    f"Top liked comment ({top_comment['likes']:,} likes): \"{top_comment['text'][:150]}...\" "
                    f"(sentiment: {top_comment['sentiment']:+.2f})",
                    styles['Normal']
                ))
            
            story.append(Spacer(1, 0.15*inch))
        
        return story
    
    def _create_sentiment_section(self, data: dict) -> list:
        """Create sentiment breakdown section"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("EMOTION DISTRIBUTION", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Create emotion chart
        chart_path = self._generate_emotion_chart(data)
        if chart_path and chart_path.exists():
            img = RLImage(str(chart_path), width=6*inch, height=3.5*inch)
            story.append(img)
        
        return story
    
    def _create_storylines_section(self, data: dict) -> list:
        """Create storylines section with detailed context"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("STORY BEATS", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Enhance storylines with LLM context if available
        enhanced_storylines = self.extract_storylines_with_context(data)
        
        if enhanced_storylines:
            for storyline in enhanced_storylines[:4]:  # Top 4 story beats
                # Urgency indicator and headline
                urgency = storyline.get('urgency', '').upper()
                headline = storyline.get('headline', storyline.get('title', storyline.get('storyline', 'Unknown')))
                
                # Map urgency to emoji/indicator
                urgency_indicator = ""
                if urgency == "HIGH ACTIVITY":
                    urgency_indicator = "🔥 HIGH ACTIVITY"
                elif urgency == "EMERGING":
                    urgency_indicator = "⚠️ EMERGING"
                elif urgency == "DEVELOPING":
                    urgency_indicator = "📊 DEVELOPING"
                else:
                    urgency_indicator = urgency if urgency else "📊 DEVELOPING"
                
                title_style = ParagraphStyle(
                    'StoryBeatTitle',
                    parent=styles['Heading2'],
                    fontSize=11,
                    textColor=colors.HexColor('#1a1a1a'),
                    spaceAfter=4
                )
                title_text = f"{urgency_indicator}: {headline}"
                story.append(Paragraph(title_text, title_style))
                
                # Description (facts only - no ET actions)
                if storyline.get('description'):
                    desc_style = ParagraphStyle(
                        'StoryBeatDesc',
                        parent=styles['Normal'],
                        fontSize=10,
                        leftIndent=0.2*inch,
                        spaceAfter=8
                    )
                    # Remove any ET action language if present
                    description = storyline['description']
                    # Clean up any prescriptive language
                    description = description.replace('ET should', '').replace('ET must', '')
                    description = description.replace('Recommendation:', '').replace('Action:', '')
                    description = description.replace('Suggestion:', '').strip()
                    story.append(Paragraph(description, desc_style))
                
                story.append(Spacer(1, 0.2*inch))
        else:
            # Fallback to simple table if LLM enhancement failed
            storylines = data.get('top_storylines', [])[:8]
            if storylines:
                table_data = [['Storyline', 'Mentions', '% of Total']]
                
                for sl in storylines:
                    table_data.append([
                        sl['storyline'].title(),
                        f"{sl['mention_count']:,}",
                        f"{sl['percentage']:.1f}%"
                    ])
                
                t = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 0), (-1, -1), 10)
                ]))
                
                story.append(t)
            else:
                story.append(Paragraph("No significant storylines detected in this period.", styles['Normal']))
        
        return story
    
    def _generate_entity_chart(self, data: dict) -> Path:
        """Generate bar chart of top entities"""
        sentiment_summary = data['sentiment_summary']
        sorted_entities = sorted(
            sentiment_summary.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:10]
        
        entities = [e[0][:25] for e in sorted_entities]
        mentions = [e[1]['total_mentions'] for e in sorted_entities]
        sentiments = [e[1]['avg_sentiment'] for e in sorted_entities]
        
        # Create color map based on sentiment
        colors_map = ['#2ecc71' if s > 0.3 else '#e74c3c' if s < -0.3 else '#95a5a6' for s in sentiments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(entities, mentions, color=colors_map)
        ax.set_xlabel('Total Mentions')
        ax.set_title('Top 10 Entities by Volume')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "entity_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _generate_emotion_chart(self, data: dict) -> Path:
        """Generate pie chart of emotion distribution"""
        # Aggregate all emotions
        all_emotions = {}
        for entity, entity_data in data['sentiment_summary'].items():
            if entity_data['emotion_breakdown']:
                for emotion, count in entity_data['emotion_breakdown'].items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + count
        
        if not all_emotions:
            return None
            
        emotions = list(all_emotions.keys())
        counts = list(all_emotions.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)
        ax.set_title('Overall Emotion Distribution')
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "emotion_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.5:
            return "Very Positive"
        elif score > 0.2:
            return "Positive"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"
