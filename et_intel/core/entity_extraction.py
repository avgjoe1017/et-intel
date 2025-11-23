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

class EntityExtractor:
    """
    Extracts and tracks entities (people, shows, storylines) from comments
    """
    
    def __init__(self):
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
        intended_subjects = self._extract_from_subjects(df)
        
        # Extract from comment text (organic mentions)
        organic_entities = self._extract_from_text(df)
        
        # Merge and deduplicate
        all_people = self._merge_entities(
            intended_subjects['people'],
            organic_entities['people']
        )
        
        all_shows = self._merge_entities(
            intended_subjects['shows'],
            organic_entities['shows']
        )
        
        # Detect couples/relationships
        couples = self._detect_relationships(df, all_people)
        
        # Detect storylines (topics that persist across multiple posts)
        storylines = self._detect_storylines(df)
        
        all_entities['people'] = all_people
        all_entities['shows'] = all_shows
        all_entities['couples'] = couples
        all_entities['storylines'] = storylines
        
        return all_entities
    
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
    
    def _extract_people_from_text(self, text: str) -> List[str]:
        """Extract all person names from comment text"""
        people = []
        
        # Match capitalized names (First Last)
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        people.extend(matches)
        
        # Also check against known entities
        for known in self.known_entities.get('people', []):
            if known.lower() in text.lower():
                people.append(known)
        
        return list(set(people))
    
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
    
    def _merge_entities(self, intended: List, organic: List) -> List:
        """Merge intended subjects with organic mentions, deduplicate"""
        merged = {}
        
        for entity, count, source in intended:
            if entity not in merged:
                merged[entity] = {'count': count, 'intended': True, 'organic': False}
        
        for entity, count, source in organic:
            if entity in merged:
                merged[entity]['count'] += count
                merged[entity]['organic'] = True
            else:
                merged[entity] = {'count': count, 'intended': False, 'organic': True}
        
        # Convert to list format
        result = [
            (name, data['count'], data['intended'], data['organic'])
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
        
        # Update known entities
        for person, _, _, _ in entities.get('people', []):
            if person not in self.known_entities['people']:
                self.known_entities['people'].append(person)
        
        for show, _, _, _ in entities.get('shows', []):
            if show not in self.known_entities['shows']:
                self.known_entities['shows'].append(show)
        
        with open(db_path, 'w') as f:
            json.dump(self.known_entities, f, indent=2)
        
        print(f"✓ Updated entity database with {len(entities['people'])} people, {len(entities['shows'])} shows")



