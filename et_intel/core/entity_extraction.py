"""
Entity Extraction Engine
Auto-detects celebrities, shows, couples, and storylines from comments
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
from .. import config
import json
from .logging_config import get_logger

logger = get_logger(__name__)

class EntityExtractor:
    """
    Extracts and tracks entities (people, shows, storylines) from comments
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Args:
            use_spacy: If True, uses spaCy NER for better entity detection
        """
        self.use_spacy = use_spacy
        self.seed_relationships = config.SEED_RELATIONSHIPS
        self.entity_cache = defaultdict(int)
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
        
        # Common celebrity name patterns
        self.celebrity_indicators = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+\b(?=\s+(?:said|wore|dated|married|starred))',  # Context clues
        ]
        
        # Show/franchise patterns
        self.show_patterns = [
            r'\b(?:The\s+)?[A-Z][a-zA-Z\s]+(?:Show|Series|Movie|Film)\b',
            r'\b[A-Z][a-zA-Z\s]+(?:\s+Season\s+\d+)?\b',
        ]
        
        # Load any previously saved entity knowledge
        self.known_entities = self._load_entity_database()
        
        # Initialize spaCy (lazy loading)
        self.nlp = None
        if self.use_spacy:
            self._init_spacy()
    
    def extract_entities_from_comments(self, df: pd.DataFrame) -> Dict:
        """
        Extract all entities from a DataFrame of comments
        
        Returns:
            Dict with:
            - 'people': List of (name, count, sentiment_avg)
            - 'shows': List of (show_name, count, sentiment_avg)
            - 'couples': List of (person1, person2, co_occurrence_count)
            - 'storylines': Detected storylines with keywords
        """
        all_entities = {
            'people': [],
            'shows': [],
            'couples': [],
            'storylines': []
        }
        
        # Extract from post subjects first (these are the "intended" subjects)
        # These represent ALL comments on that post (implicit mentions)
        intended_subjects = self._extract_from_subjects(df)
        
        # Extract from comment text (explicit mentions in comments)
        organic_entities = self._extract_from_text(df)
        
        # Count implicit mentions (comments on posts about entities)
        implicit_counts = self._count_implicit_mentions(df, intended_subjects)
        
        # Merge and deduplicate (including implicit mentions)
        all_people = self._merge_entities(
            intended_subjects['people'],
            organic_entities['people'],
            implicit_counts
        )
        
        all_shows = self._merge_entities(
            intended_subjects['shows'],
            organic_entities['shows'],
            implicit_counts
        )
        
        # Detect couples/relationships
        couples = self._detect_relationships(df, all_people)
        
        # Detect storylines (topics that persist across multiple posts)
        storylines = self._detect_storylines(df)
        
        all_entities['people'] = all_people
        all_entities['shows'] = all_shows
        all_entities['couples'] = couples
        all_entities['storylines'] = storylines
        
        # Canonicalize entities (deduplicate)
        all_entities = self.canonicalize_entities(all_entities)
        
        # LLM Enhancement: Further canonicalization with GPT (optional)
        if config.USE_LLM_ENHANCEMENT and config.OPENAI_API_KEY:
            try:
                all_entities = self.canonicalize_with_gpt(all_entities)
            except Exception as e:
                logger.warning(f"LLM canonicalization failed, using rule-based results: {e}")
        
        return all_entities
    
    def canonicalize_entities(self, entities_dict):
        """
        Merge duplicate entities with fuzzy matching
        Remove garbage entities
        """
        from difflib import SequenceMatcher
        
        # Remove garbage entities
        GARBAGE_PATTERNS = [
            'dataset instagram',
            'instagram dataset',
            'the bible',  # Unless actually relevant to content
            'http',
            'www.',
        ]
        
        people = entities_dict.get('people', [])
        
        # Filter garbage
        people = [p for p in people if not any(
            garbage.lower() in p[0].lower() 
            for garbage in GARBAGE_PATTERNS
        )]
        
        # Team name mappings (merge team names with their associated people)
        TEAM_MAPPINGS = {
            'team justin': 'Justin Baldoni',
            'team blake': 'Blake Lively',
            'team blake lively': 'Blake Lively',
            'team justin baldoni': 'Justin Baldoni',
        }
        
        # First pass: merge team names with their people
        merged_people = []
        for entity in people:
            name = entity[0]
            name_lower = name.lower()
            if name_lower in TEAM_MAPPINGS:
                # Replace team name with canonical person name
                entity_list = list(entity)
                entity_list[0] = TEAM_MAPPINGS[name_lower]
                merged_people.append(tuple(entity_list))
            else:
                merged_people.append(entity)
        people = merged_people
        
        # Canonicalize (merge similar names)
        canonical_map = {}
        
        for entity in people:
            name = entity[0]
            name_lower = name.lower()
            
            # Check if this is similar to existing canonical entity
            found_match = False
            for canonical_name in list(canonical_map.keys()):
                canonical_lower = canonical_name.lower()
                
                # Check for partial name matches (e.g., "Blake" in "Blake Lively")
                # If one name is a substring of another and both are reasonable length, merge
                is_partial_match = False
                if len(name) >= 3 and len(canonical_name) >= 3:
                    if name_lower in canonical_lower or canonical_lower in name_lower:
                        # Prefer the longer/more complete name
                        is_partial_match = True
                
                similarity = SequenceMatcher(None, name_lower, canonical_lower).ratio()
                
                # If >80% similar OR partial match, merge
                if similarity > 0.8 or is_partial_match:
                    # Use the longer/more complete name as canonical
                    if len(name) > len(canonical_name):
                        # Update canonical
                        old_data = canonical_map[canonical_name]
                        # Sum mentions (index 1) and implicit counts (index 4 if exists)
                        new_count = old_data[1] + entity[1]
                        new_implicit = (old_data[4] if len(old_data) > 4 else 0) + (entity[4] if len(entity) > 4 else 0)
                        
                        # Create merged entity tuple
                        merged_entity = list(entity)
                        merged_entity[1] = new_count
                        if len(merged_entity) > 4:
                            merged_entity[4] = new_implicit
                        elif len(merged_entity) == 4: # Convert old format to new if needed
                             merged_entity.append(new_implicit)
                             
                        del canonical_map[canonical_name]
                        canonical_map[name] = merged_entity
                    else:
                        # Add to existing
                        canonical_entity = list(canonical_map[canonical_name])
                        canonical_entity[1] += entity[1] # Sum mentions
                        if len(canonical_entity) > 4 and len(entity) > 4:
                            canonical_entity[4] += entity[4]
                        canonical_map[canonical_name] = canonical_entity
                        
                    found_match = True
                    break
            
            if not found_match:
                canonical_map[name] = list(entity)
        
        # Convert back to list format and sort
        entities_dict['people'] = sorted(list(canonical_map.values()), key=lambda x: x[1], reverse=True)
        
        return entities_dict
    
    def canonicalize_with_gpt(self, entities_dict):
        """
        Use GPT-4o-mini to further canonicalize entities after rule-based deduplication
        Merges duplicates and removes garbage entities intelligently
        
        Cost: ~$0.001 per brief
        """
        if not config.OPENAI_API_KEY or not config.OPENAI_API_KEY.strip():
            logger.warning("OpenAI API key not available for LLM canonicalization")
            return entities_dict
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client for canonicalization: {e}")
            return entities_dict
        
        # Extract entity names from people list
        people_names = [entity[0] for entity in entities_dict.get('people', [])]
        
        if not people_names:
            return entities_dict
        
        # Limit to top 50 entities to manage costs
        people_names = people_names[:50]
        
        prompt = f"""These entities were extracted from Instagram comments about entertainment news. 
Merge duplicates (e.g., "Blake" and "Blake Lively" should become "Blake Lively") and remove garbage entities (like "Dataset Instagram", "the Bible" unless actually relevant).

IMPORTANT: Merge team names with their associated people:
- "Team Justin" or "team justin" → "Justin Baldoni"
- "Team Blake" or "team blake" → "Blake Lively"

Entities: {json.dumps(people_names, indent=2)}

Return JSON with only the canonical entity names (no duplicates, no garbage, no team names):
{{"canonical_entities": ["Name1", "Name2", ...]}}"""

        try:
            response = client.chat.completions.create(
                model=config.ENTITY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying and canonicalizing celebrity and entertainment entity names."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=config.MAX_TOKENS_ENTITY
            )
            
            result = json.loads(response.choices[0].message.content)
            canonical_names = result.get('canonical_entities', [])
            
            # Map original entities to canonical names (more lenient matching)
            canonical_set = set(n.lower() for n in canonical_names)
            filtered_entities = []
            seen_canonical = set()
            entity_to_canonical = {}  # Track mappings
            
            # First pass: exact matches
            for entity in entities_dict.get('people', []):
                entity_name = entity[0]
                entity_lower = entity_name.lower()
                
                # Check if this entity matches a canonical name exactly
                if entity_lower in canonical_set:
                    canonical_match = next((n for n in canonical_names if n.lower() == entity_lower), entity_name)
                    entity_to_canonical[entity_name] = canonical_match
            
            # Second pass: partial/fuzzy matches for unmatched entities
            for entity in entities_dict.get('people', []):
                entity_name = entity[0]
                entity_lower = entity_name.lower()
                
                if entity_name in entity_to_canonical:
                    # Already matched, skip
                    continue
                
                # Try partial matching
                best_match = None
                best_score = 0
                for canonical in canonical_names:
                    can_lower = canonical.lower()
                    # Check if one contains the other (for "Blake" -> "Blake Lively")
                    if can_lower in entity_lower or entity_lower in can_lower:
                        # Prefer longer/more complete name
                        score = len(canonical)
                        if score > best_score:
                            best_match = canonical
                            best_score = score
                
                if best_match:
                    entity_to_canonical[entity_name] = best_match
                else:
                    # Keep original if no match found
                    entity_to_canonical[entity_name] = entity_name
            
            # Build filtered list with canonical names
            for entity in entities_dict.get('people', []):
                entity_name = entity[0]
                canonical_name = entity_to_canonical.get(entity_name, entity_name)
                canonical_lower = canonical_name.lower()
                
                if canonical_lower not in seen_canonical:
                    # Update entity name to canonical form
                    updated_entity = list(entity)
                    updated_entity[0] = canonical_name
                    filtered_entities.append(tuple(updated_entity))
                    seen_canonical.add(canonical_lower)
                else:
                    # Merge with existing canonical entity (sum counts)
                    for i, existing in enumerate(filtered_entities):
                        if existing[0].lower() == canonical_lower:
                            merged = list(existing)
                            merged[1] += entity[1]  # Sum mentions
                            if len(merged) > 4 and len(entity) > 4:
                                merged[4] += entity[4]  # Sum implicit counts
                            filtered_entities[i] = tuple(merged)
                            break
            
            # Safety check: if we lost more than 30% of entities, something went wrong
            if len(filtered_entities) < len(entities_dict.get('people', [])) * 0.7:
                logger.warning(f"LLM canonicalization reduced entities from {len(entities_dict.get('people', []))} to {len(filtered_entities)}, using original list")
                return entities_dict
            
            entities_dict['people'] = sorted(filtered_entities, key=lambda x: x[1], reverse=True)
            logger.info(f"LLM canonicalization: {len(people_names)} -> {len(canonical_names)} entities")
            
        except Exception as e:
            logger.warning(f"LLM canonicalization failed: {e}. Using rule-based results.")
            return entities_dict
        
        return entities_dict

    def _extract_from_subjects(self, df: pd.DataFrame) -> Dict:
        """Extract entities from post subjects/titles"""
        entities = {'people': [], 'shows': []}
        
        subjects = df['post_subject'].dropna().unique()
        
        for subject in subjects:
            # Check if it's a person (has first + last name pattern)
            if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', subject):
                person_name = self._extract_person_name(subject)
                if person_name:
                    count = len(df[df['post_subject'] == subject])
                    entities['people'].append((person_name, count, 'intended_subject'))
            
            # Check if it's a show/movie
            if any(keyword in subject.lower() for keyword in ['show', 'movie', 'series', 'season']):
                count = len(df[df['post_subject'] == subject])
                entities['shows'].append((subject, count, 'intended_subject'))
        
        return entities
    
    def _extract_from_text(self, df: pd.DataFrame) -> Dict:
        """Extract entities from comment text using NER patterns"""
        entities = {'people': Counter(), 'shows': Counter()}
        
        for comment in df['comment_text']:
            # Extract person names
            people = self._extract_people_from_text(comment)
            for person in people:
                entities['people'][person] += 1
            
            # Extract show names
            shows = self._extract_shows_from_text(comment)
            for show in shows:
                entities['shows'][show] += 1
        
        # Filter by minimum mention threshold
        min_mentions = config.MIN_ENTITY_MENTIONS
        
        people_list = [
            (name, count, 'organic_mention') 
            for name, count in entities['people'].items() 
            if count >= min_mentions
        ]
        
        shows_list = [
            (name, count, 'organic_mention') 
            for name, count in entities['shows'].items() 
            if count >= min_mentions
        ]
        
        return {'people': people_list, 'shows': shows_list}
    
    def _extract_person_name(self, text: str) -> str:
        """Extract a person's name from text"""
        # Match First Last name pattern
        match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
        return match.group(1) if match else None
    
    def _init_spacy(self):
        """Initialize spaCy NER model (lazy loading)"""
        try:
            import spacy
            logger = __import__('logging').getLogger(__name__)
            
            # Try to load the large model, fallback to medium or small
            try:
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("[OK] Loaded spaCy en_core_web_lg model")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("[OK] Loaded spaCy en_core_web_md model")
                except OSError:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("[OK] Loaded spaCy en_core_web_sm model")
        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.warning(f"spaCy not available: {e}. Install with: python -m spacy download en_core_web_lg")
            self.nlp = None
            self.use_spacy = False
    
    def _extract_people_from_text(self, text: str) -> List[str]:
        """Extract all person names from comment text"""
        people = []
        
        # Validate input
        if not text or not isinstance(text, str):
            return people
        
        # Use spaCy NER if available (much more accurate)
        if self.use_spacy and self.nlp:
            try:
                # Limit text length to avoid memory issues
                text_limited = text[:1000] if len(text) > 1000 else text
                doc = self.nlp(text_limited)
                # Extract PERSON entities
                spacy_people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                people.extend(spacy_people)
                
                # Also extract from noun chunks that look like names
                for chunk in doc.noun_chunks:
                    try:
                        if chunk.root.pos_ == "PROPN" and len(chunk.text.split()) >= 2:
                            # Likely a person name (proper noun with 2+ words)
                            if chunk.text not in spacy_people:
                                people.append(chunk.text)
                    except Exception as e:
                        logger = __import__('logging').getLogger(__name__)
                        logger.debug(f"Error processing noun chunk: {e}")
            except Exception as e:
                logger = __import__('logging').getLogger(__name__)
                logger.debug(f"spaCy extraction error: {e}")
        
        # Fallback to regex patterns
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        people.extend(matches)
        
        # Also check against known entities
        for known in self.known_entities.get('people', []):
            if known.lower() in text.lower():
                people.append(known)
        
        # Clean and deduplicate
        cleaned = []
        for person in people:
            person = person.strip()
            if len(person) > 2 and person not in cleaned:
                cleaned.append(person)
        
        return cleaned
    
    def _extract_shows_from_text(self, text: str) -> List[str]:
        """Extract show/movie names from text"""
        shows = []
        
        # Check known shows first
        for known in self.known_entities.get('shows', []):
            if known.lower() in text.lower():
                shows.append(known)
        
        # Pattern matching for show names
        matches = re.findall(r'\b([A-Z][a-zA-Z\s]+(?:Show|Series|Movie|Film))\b', text)
        shows.extend(matches)
        
        return list(set(shows))
    
    def _detect_relationships(self, df: pd.DataFrame, people_list: List) -> List[Tuple]:
        """
        Detect couples/relationships based on co-occurrence
        
        Uses both seed list and auto-discovery
        """
        couples = []
        
        # Extract person names from people_list
        people_names = [p[0] for p in people_list]
        
        # Build co-occurrence matrix
        for comment in df['comment_text']:
            mentioned = [p for p in people_names if p.lower() in comment.lower()]
            
            # Track co-occurrences
            for i, person1 in enumerate(mentioned):
                for person2 in mentioned[i+1:]:
                    pair = tuple(sorted([person1, person2]))
                    self.co_occurrence_matrix[pair[0]][pair[1]] += 1
        
        # Check seed relationships
        for pair in self.seed_relationships:
            person1, person2 = pair[0], pair[1]
            if person1 in people_names and person2 in people_names:
                co_count = self.co_occurrence_matrix[person1].get(person2, 0)
                if co_count > 0:
                    couples.append((person1, person2, co_count, 'seed'))
        
        # Auto-discover new relationships
        for person1 in people_names:
            for person2, co_count in self.co_occurrence_matrix[person1].items():
                if person2 in people_names:
                    # Calculate co-occurrence rate
                    person1_total = sum(1 for c in df['comment_text'] if person1.lower() in c.lower())
                    person2_total = sum(1 for c in df['comment_text'] if person2.lower() in c.lower())
                    
                    if person1_total > 0 and person2_total > 0:
                        co_rate = co_count / min(person1_total, person2_total)
                        
                        if co_rate >= config.COUPLE_THRESHOLD:
                            # Check if not already in seed list
                            pair_sorted = tuple(sorted([person1, person2]))
                            is_seed = any(
                                set([person1, person2]) == set(seed) 
                                for seed in self.seed_relationships
                            )
                            if not is_seed:
                                couples.append((person1, person2, co_count, 'discovered'))
        
        return couples
    
    def _detect_storylines(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect persistent storylines/topics
        (e.g., "lawsuit", "breakup", "pregnancy rumors")
        """
        storyline_keywords = {
            'lawsuit': ['lawsuit', 'legal', 'court', 'suing', 'sue'],
            'relationship': ['dating', 'relationship', 'couple', 'romance', 'together'],
            'breakup': ['breakup', 'split', 'divorce', 'separated', 'broken up'],
            'pregnancy': ['pregnant', 'baby', 'expecting', 'pregnancy'],
            'controversy': ['scandal', 'controversy', 'backlash', 'drama', 'accused'],
            'career': ['new role', 'cast', 'movie', 'project', 'filming', 'album'],
        }
        
        storylines = []
        
        for storyline_name, keywords in storyline_keywords.items():
            count = 0
            for comment in df['comment_text']:
                if any(kw in comment.lower() for kw in keywords):
                    count += 1
            
            if count >= config.MIN_ENTITY_MENTIONS * 2:  # Higher threshold for storylines
                storylines.append({
                    'storyline': storyline_name,
                    'mention_count': count,
                    'percentage': count / len(df) * 100
                })
        
        return sorted(storylines, key=lambda x: x['mention_count'], reverse=True)
    
    def _count_implicit_mentions(self, df: pd.DataFrame, intended_subjects: Dict) -> Dict:
        """
        Count implicit mentions - comments on posts about entities
        These are comments that don't explicitly name the entity but are clearly about them
        """
        implicit_counts = {'people': {}, 'shows': {}}
        
        # For each post subject, count all comments on that post
        for subject in df['post_subject'].dropna().unique():
            if not subject:
                continue
            
            # Get all comments on posts with this subject
            post_comments = df[df['post_subject'] == subject]
            comment_count = len(post_comments)
            
            # Extract entities from subject
            # Check if subject contains person names
            for person_tuple in intended_subjects.get('people', []):
                if isinstance(person_tuple, (tuple, list)) and len(person_tuple) > 0:
                    person = person_tuple[0]
                    if person.lower() in subject.lower():
                        if person not in implicit_counts['people']:
                            implicit_counts['people'][person] = 0
                        implicit_counts['people'][person] += comment_count
            
            # Check if subject contains show names
            for show_tuple in intended_subjects.get('shows', []):
                if isinstance(show_tuple, (tuple, list)) and len(show_tuple) > 0:
                    show = show_tuple[0]
                    if show.lower() in subject.lower():
                        if show not in implicit_counts['shows']:
                            implicit_counts['shows'][show] = 0
                        implicit_counts['shows'][show] += comment_count
        
        return implicit_counts
    
    def _merge_entities(self, intended: List, organic: List, implicit_counts: Dict = None) -> List:
        """Merge intended subjects with organic mentions, including implicit counts"""
        merged = {}
        implicit_counts = implicit_counts or {'people': {}, 'shows': {}}
        
        # Start with intended subjects (these get implicit mention counts)
        for entity_tuple in intended:
            if isinstance(entity_tuple, (tuple, list)) and len(entity_tuple) >= 3:
                entity = entity_tuple[0]
                count = entity_tuple[1] if len(entity_tuple) > 1 else 0
                
                if entity not in merged:
                    # Get implicit count (all comments on posts about this entity)
                    implicit_count = implicit_counts.get('people', {}).get(entity, 0)
                    if implicit_count == 0:
                        implicit_count = implicit_counts.get('shows', {}).get(entity, 0)
                    
                    merged[entity] = {
                        'count': count,  # Explicit mentions in subject
                        'implicit_count': implicit_count,  # All comments on posts about this
                        'intended': True, 
                        'organic': False
                    }
        
        # Add organic mentions
        for entity_tuple in organic:
            if isinstance(entity_tuple, (tuple, list)) and len(entity_tuple) >= 2:
                entity = entity_tuple[0]
                count = entity_tuple[1] if len(entity_tuple) > 1 else 0
                
                if entity in merged:
                    merged[entity]['count'] += count  # Add explicit mentions
                    merged[entity]['organic'] = True
                else:
                    merged[entity] = {
                        'count': count,
                        'implicit_count': 0,
                        'intended': False, 
                        'organic': True
                    }
        
        # Convert to list format: (name, total_count, intended, organic, implicit_count)
        result = [
            (name, data['count'] + data['implicit_count'], data['intended'], data['organic'], data['implicit_count'])
            for name, data in merged.items()
        ]
        
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    def _load_entity_database(self) -> Dict:
        """Load previously identified entities (to improve future recognition)"""
        db_path = config.DB_DIR / "known_entities.json"
        
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        
        # Default seed entities (expand this over time)
        return {
            'people': [
                'Taylor Swift', 'Travis Kelce', 'Blake Lively', 'Justin Baldoni',
                'Timothée Chalamet', 'Ariana Grande', 'Jennifer Lopez', 'Ben Affleck'
            ],
            'shows': [
                'It Ends With Us', 'The Late Show', 'Entertainment Tonight',
                'Jeopardy', 'WOW: Women of Wrestling'
            ]
        }
    
    def save_entity_database(self, entities: Dict):
        """Save discovered entities for future use"""
        db_path = config.DB_DIR / "known_entities.json"
        
        # Update known entities (handle both old and new tuple formats)
        for entity_tuple in entities.get('people', []):
            if isinstance(entity_tuple, (tuple, list)) and len(entity_tuple) > 0:
                person = entity_tuple[0]
                if person not in self.known_entities['people']:
                    self.known_entities['people'].append(person)
        
        for entity_tuple in entities.get('shows', []):
            if isinstance(entity_tuple, (tuple, list)) and len(entity_tuple) > 0:
                show = entity_tuple[0]
                if show not in self.known_entities['shows']:
                    self.known_entities['shows'].append(show)
        
        with open(db_path, 'w') as f:
            json.dump(self.known_entities, f, indent=2)
        
        # Logging will be handled by caller if needed



