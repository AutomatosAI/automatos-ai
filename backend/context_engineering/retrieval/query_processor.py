
"""
Query Processing and Expansion System
====================================

Advanced query understanding, expansion, and processing for intelligent context retrieval.
Handles query intent classification, semantic expansion, and query optimization.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

# Import mathematical foundations
from ..mathematical_foundations.information_theory import InformationTheory
from ..mathematical_foundations.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of query intents"""
    FACTUAL = "factual"           # Looking for facts or data
    PROCEDURAL = "procedural"     # How-to or step-by-step
    CONCEPTUAL = "conceptual"     # Understanding concepts
    COMPARATIVE = "comparative"   # Comparing things
    CAUSAL = "causal"            # Cause and effect
    TEMPORAL = "temporal"        # Time-related queries
    SPATIAL = "spatial"          # Location-related queries
    EXPLORATORY = "exploratory"  # Open-ended exploration

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"         # Single concept, direct
    MODERATE = "moderate"     # Multiple concepts, some complexity
    COMPLEX = "complex"       # Many concepts, high complexity
    MULTI_PART = "multi_part" # Multiple distinct sub-queries

@dataclass
class QueryEntity:
    """An entity extracted from a query"""
    text: str
    entity_type: str  # person, organization, location, concept, etc.
    confidence: float
    importance: float  # How important this entity is to the query

@dataclass
class QueryConcept:
    """A concept identified in a query"""
    text: str
    concept_type: str  # technical, business, general, etc.
    related_terms: List[str]
    importance: float

@dataclass
class ProcessedQuery:
    """A fully processed and expanded query"""
    original_text: str
    cleaned_text: str
    intent: QueryIntent
    complexity: QueryComplexity
    entities: List[QueryEntity]
    concepts: List[QueryConcept]
    keywords: List[str]
    expanded_terms: List[str]
    synonyms: Dict[str, List[str]]
    sub_queries: List[str]  # For complex queries
    context_hints: List[str]  # Additional context clues
    embedding_ready_text: str  # Optimized for embedding
    search_terms: List[str]  # Optimized for search
    filters: Dict[str, Any]  # Suggested filters
    confidence_score: float

class QueryProcessor:
    """Advanced query processing and expansion system"""
    
    def __init__(self):
        # Mathematical components
        self.info_theory = InformationTheory()
        self.stats = StatisticalAnalysis()
        
        # Knowledge bases
        self.synonym_dict = self._load_synonyms()
        self.entity_patterns = self._load_entity_patterns()
        self.intent_patterns = self._load_intent_patterns()
        self.concept_dictionary = self._load_concept_dictionary()
        
        # Query processing statistics
        self.processing_stats = {
            'total_queries': 0,
            'intent_distribution': {},
            'complexity_distribution': {},
            'avg_expansion_ratio': 0.0
        }
    
    def process_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """Main query processing method"""
        
        self.processing_stats['total_queries'] += 1
        
        # Step 1: Clean and normalize the query
        cleaned_text = self._clean_query(query_text)
        
        # Step 2: Identify intent
        intent = self._identify_intent(cleaned_text)
        
        # Step 3: Assess complexity
        complexity = self._assess_complexity(cleaned_text)
        
        # Step 4: Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Step 5: Identify concepts
        concepts = self._identify_concepts(cleaned_text, entities)
        
        # Step 6: Extract keywords
        keywords = self._extract_keywords(cleaned_text)
        
        # Step 7: Expand query terms
        expanded_terms = self._expand_terms(keywords, concepts)
        
        # Step 8: Generate synonyms
        synonyms = self._generate_synonyms(keywords + [c.text for c in concepts])
        
        # Step 9: Handle complex queries
        sub_queries = self._decompose_query(cleaned_text, complexity)
        
        # Step 10: Extract context hints
        context_hints = self._extract_context_hints(cleaned_text, context)
        
        # Step 11: Optimize for embedding
        embedding_text = self._optimize_for_embedding(cleaned_text, expanded_terms)
        
        # Step 12: Optimize for search
        search_terms = self._optimize_for_search(keywords, expanded_terms)
        
        # Step 13: Generate filters
        filters = self._generate_filters(entities, concepts, intent, context)
        
        # Step 14: Calculate confidence
        confidence = self._calculate_confidence(entities, concepts, keywords)
        
        # Update statistics
        self._update_statistics(intent, complexity, len(expanded_terms) / max(len(keywords), 1))
        
        return ProcessedQuery(
            original_text=query_text,
            cleaned_text=cleaned_text,
            intent=intent,
            complexity=complexity,
            entities=entities,
            concepts=concepts,
            keywords=keywords,
            expanded_terms=expanded_terms,
            synonyms=synonyms,
            sub_queries=sub_queries,
            context_hints=context_hints,
            embedding_ready_text=embedding_text,
            search_terms=search_terms,
            filters=filters,
            confidence_score=confidence
        )
    
    def _clean_query(self, query_text: str) -> str:
        """Clean and normalize query text"""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query_text.strip())
        
        # Handle common contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            cleaned = cleaned.replace(contraction, expansion)
        
        # Normalize punctuation
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _identify_intent(self, query_text: str) -> QueryIntent:
        """Identify the intent of the query"""
        
        query_lower = query_text.lower()
        
        # Check for intent patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return QueryIntent(intent)
        
        # Default classification based on structure
        if any(word in query_lower for word in ['how', 'step', 'process', 'method']):
            return QueryIntent.PROCEDURAL
        elif any(word in query_lower for word in ['what', 'define', 'explain', 'concept']):
            return QueryIntent.CONCEPTUAL
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return QueryIntent.COMPARATIVE
        elif any(word in query_lower for word in ['why', 'because', 'cause', 'reason']):
            return QueryIntent.CAUSAL
        elif any(word in query_lower for word in ['when', 'time', 'date', 'history']):
            return QueryIntent.TEMPORAL
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return QueryIntent.SPATIAL
        else:
            return QueryIntent.FACTUAL
    
    def _assess_complexity(self, query_text: str) -> QueryComplexity:
        """Assess the complexity of the query"""
        
        # Count various complexity indicators
        word_count = len(query_text.split())
        sentence_count = len(re.split(r'[.!?]+', query_text))
        conjunction_count = len(re.findall(r'\b(and|or|but|however|therefore|because)\b', query_text.lower()))
        question_words = len(re.findall(r'\b(what|when|where|who|why|how|which)\b', query_text.lower()))
        
        # Simple complexity scoring
        complexity_score = 0
        complexity_score += min(word_count / 10, 3)  # Word count factor
        complexity_score += min(sentence_count - 1, 2)  # Multi-sentence
        complexity_score += min(conjunction_count, 3)  # Conjunctions
        complexity_score += min(question_words / 2, 2)  # Multiple question words
        
        if complexity_score >= 6:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 4:
            return QueryComplexity.MODERATE
        elif sentence_count > 1 or conjunction_count > 2:
            return QueryComplexity.MULTI_PART
        else:
            return QueryComplexity.SIMPLE
    
    def _extract_entities(self, query_text: str) -> List[QueryEntity]:
        """Extract entities from the query"""
        
        entities = []
        
        # Use patterns to identify different entity types
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 1:  # Skip single characters
                        
                        # Calculate importance based on position and length
                        position_factor = 1.0 - (match.start() / len(query_text)) * 0.3
                        length_factor = min(len(entity_text) / 20, 1.0)
                        importance = (position_factor + length_factor) / 2
                        
                        entity = QueryEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=0.8,  # Pattern-based confidence
                            importance=importance
                        )
                        entities.append(entity)
        
        # Remove duplicates and sort by importance
        unique_entities = {e.text.lower(): e for e in entities}.values()
        return sorted(unique_entities, key=lambda x: x.importance, reverse=True)
    
    def _identify_concepts(self, query_text: str, entities: List[QueryEntity]) -> List[QueryConcept]:
        """Identify concepts in the query"""
        
        concepts = []
        query_words = set(query_text.lower().split())
        
        # Check against concept dictionary
        for concept_text, concept_data in self.concept_dictionary.items():
            # Check if concept or related terms are in query
            concept_words = set(concept_text.lower().split())
            related_words = set(word.lower() for word in concept_data.get('related', []))
            
            if concept_words.intersection(query_words) or related_words.intersection(query_words):
                # Calculate importance based on word overlap
                overlap_ratio = len(concept_words.intersection(query_words)) / len(concept_words)
                
                concept = QueryConcept(
                    text=concept_text,
                    concept_type=concept_data.get('type', 'general'),
                    related_terms=concept_data.get('related', []),
                    importance=overlap_ratio
                )
                concepts.append(concept)
        
        # Sort by importance
        return sorted(concepts, key=lambda x: x.importance, reverse=True)[:10]  # Limit to top 10
    
    def _extract_keywords(self, query_text: str) -> List[str]:
        """Extract important keywords from the query"""
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Calculate word importance using information theory
        keyword_scores = {}
        for keyword in set(keywords):
            # Simple TF in query
            tf = keywords.count(keyword)
            # Length bonus
            length_bonus = min(len(keyword) / 10, 1.0)
            # Position bonus (earlier words are more important)
            first_position = keywords.index(keyword)
            position_bonus = 1.0 - (first_position / len(keywords)) * 0.3
            
            score = tf * (1 + length_bonus + position_bonus)
            keyword_scores[keyword] = score
        
        # Return sorted keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, score in sorted_keywords]
    
    def _expand_terms(self, keywords: List[str], concepts: List[QueryConcept]) -> List[str]:
        """Expand query terms with related terms"""
        
        expanded_terms = keywords.copy()
        
        # Add concept-related terms
        for concept in concepts:
            expanded_terms.extend(concept.related_terms[:3])  # Limit to top 3 related terms
        
        # Add morphological variations
        for keyword in keywords:
            variations = self._get_morphological_variations(keyword)
            expanded_terms.extend(variations)
        
        # Add domain-specific expansions
        for keyword in keywords:
            domain_terms = self._get_domain_expansions(keyword)
            expanded_terms.extend(domain_terms)
        
        # Remove duplicates and return
        return list(set(expanded_terms))
    
    def _generate_synonyms(self, terms: List[str]) -> Dict[str, List[str]]:
        """Generate synonyms for query terms"""
        
        synonyms = {}
        
        for term in terms:
            term_lower = term.lower()
            if term_lower in self.synonym_dict:
                synonyms[term] = self.synonym_dict[term_lower][:5]  # Limit to top 5 synonyms
        
        return synonyms
    
    def _decompose_query(self, query_text: str, complexity: QueryComplexity) -> List[str]:
        """Decompose complex queries into sub-queries"""
        
        if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            return []
        
        sub_queries = []
        
        # Split by conjunctions
        conjunctions = ['and', 'or', 'but', 'however', 'therefore', 'because']
        
        current_query = query_text
        for conjunction in conjunctions:
            parts = re.split(f'\\b{conjunction}\\b', current_query, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_queries.extend([part.strip() for part in parts if part.strip()])
                break
        
        # Split by question words if multiple
        question_patterns = [r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b', r'\bwhy\b', r'\bhow\b']
        
        for pattern in question_patterns:
            matches = list(re.finditer(pattern, query_text, re.IGNORECASE))
            if len(matches) > 1:
                # Split at each question word
                positions = [match.start() for match in matches]
                for i, pos in enumerate(positions):
                    end_pos = positions[i + 1] if i + 1 < len(positions) else len(query_text)
                    sub_query = query_text[pos:end_pos].strip()
                    if sub_query:
                        sub_queries.append(sub_query)
                break
        
        return sub_queries[:5]  # Limit to 5 sub-queries
    
    def _extract_context_hints(self, query_text: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Extract context hints from query and external context"""
        
        hints = []
        
        # Extract temporal hints
        temporal_patterns = [
            r'\b(today|yesterday|tomorrow|now|currently|recent|latest)\b',
            r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{4})\b',
            r'\b(last|next|previous|this)\s+(week|month|year|quarter)\b'
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            hints.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # Extract domain hints
        domain_patterns = {
            'technical': r'\b(API|algorithm|code|programming|software|system|database)\b',
            'business': r'\b(revenue|profit|sales|marketing|strategy|customer|client)\b',
            'academic': r'\b(research|study|analysis|paper|journal|methodology)\b'
        }
        
        for domain, pattern in domain_patterns.items():
            if re.search(pattern, query_text, re.IGNORECASE):
                hints.append(f"domain:{domain}")
        
        # Add context from external source
        if context:
            if 'user_domain' in context:
                hints.append(f"user_domain:{context['user_domain']}")
            if 'previous_queries' in context:
                hints.append("has_context")
            if 'document_types' in context:
                hints.extend([f"doc_type:{doc_type}" for doc_type in context['document_types']])
        
        return list(set(hints))
    
    def _optimize_for_embedding(self, cleaned_text: str, expanded_terms: List[str]) -> str:
        """Optimize text for embedding generation"""
        
        # Combine original text with important expanded terms
        important_expansions = expanded_terms[:10]  # Top 10 expansions
        
        # Create embedding-optimized text
        optimized_parts = [cleaned_text]
        
        # Add expanded terms that aren't already in the text
        text_words = set(cleaned_text.lower().split())
        for term in important_expansions:
            if term.lower() not in text_words:
                optimized_parts.append(term)
        
        return ' '.join(optimized_parts)
    
    def _optimize_for_search(self, keywords: List[str], expanded_terms: List[str]) -> List[str]:
        """Optimize terms for search operations"""
        
        # Combine and prioritize search terms
        all_terms = keywords + expanded_terms
        
        # Score terms by importance
        term_scores = {}
        for term in all_terms:
            score = 0
            
            # Keyword bonus
            if term in keywords:
                score += 2
            
            # Length bonus
            score += min(len(term) / 10, 1)
            
            # Frequency penalty (avoid over-representation)
            frequency = all_terms.count(term)
            score /= max(frequency, 1)
            
            term_scores[term] = score
        
        # Return top scored terms
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:20]]  # Top 20 search terms
    
    def _generate_filters(
        self, 
        entities: List[QueryEntity], 
        concepts: List[QueryConcept], 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate suggested filters for the search"""
        
        filters = {}
        
        # Entity-based filters
        for entity in entities:
            if entity.entity_type == 'organization':
                filters.setdefault('sources', []).append(entity.text.lower())
            elif entity.entity_type == 'person':
                filters.setdefault('authors', []).append(entity.text.lower())
            elif entity.entity_type == 'date':
                # This would need more sophisticated date parsing
                filters['has_temporal_filter'] = True
        
        # Intent-based filters
        if intent == QueryIntent.PROCEDURAL:
            filters.setdefault('document_types', []).append('tutorial')
            filters.setdefault('document_types', []).append('guide')
        elif intent == QueryIntent.FACTUAL:
            filters.setdefault('document_types', []).append('reference')
            filters.setdefault('document_types', []).append('documentation')
        
        # Concept-based filters
        for concept in concepts:
            if concept.concept_type == 'technical':
                filters.setdefault('categories', []).append('technical')
            elif concept.concept_type == 'business':
                filters.setdefault('categories', []).append('business')
        
        # Context-based filters
        if context:
            if 'preferred_sources' in context:
                filters.setdefault('sources', []).extend(context['preferred_sources'])
            if 'time_range' in context:
                filters['time_range'] = context['time_range']
        
        return filters
    
    def _calculate_confidence(
        self, 
        entities: List[QueryEntity], 
        concepts: List[QueryConcept], 
        keywords: List[str]
    ) -> float:
        """Calculate confidence in query processing"""
        
        # Base confidence from entity and concept recognition
        entity_confidence = sum(e.confidence for e in entities) / max(len(entities), 1)
        concept_confidence = sum(c.importance for c in concepts) / max(len(concepts), 1)
        
        # Keyword richness
        keyword_richness = min(len(keywords) / 5, 1.0)  # Normalize to 5 keywords
        
        # Overall confidence
        confidence = (entity_confidence * 0.4 + concept_confidence * 0.4 + keyword_richness * 0.2)
        
        return min(confidence, 1.0)
    
    # Helper methods for loading knowledge bases
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary"""
        # This would typically load from a file or database
        return {
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'miniature', 'compact'],
            'good': ['excellent', 'great', 'wonderful', 'fantastic'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'make': ['create', 'build', 'construct', 'develop'],
            'show': ['display', 'demonstrate', 'present', 'exhibit'],
            'find': ['locate', 'discover', 'identify', 'search'],
            'use': ['utilize', 'employ', 'apply', 'implement']
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load entity recognition patterns"""
        return {
            'organization': [
                r'\b[A-Z][a-zA-Z]*\s+(Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b(Google|Microsoft|Amazon|Apple|Facebook|IBM|Oracle)\b'
            ],
            'person': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                r'\bDr\.\s+[A-Z][a-z]+\b',
                r'\bProf\.\s+[A-Z][a-z]+\b'
            ],
            'technology': [
                r'\b(API|REST|GraphQL|SQL|NoSQL|AI|ML|NLP|GPU|CPU)\b',
                r'\b(Python|JavaScript|Java|C\+\+|React|Angular|Vue)\b'
            ],
            'date': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
        }
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent classification patterns"""
        return {
            'procedural': [
                r'\bhow\s+to\b',
                r'\bsteps?\s+to\b',
                r'\bprocess\s+of\b',
                r'\bmethod\s+for\b'
            ],
            'factual': [
                r'\bwhat\s+is\b',
                r'\bdefine\b',
                r'\bmean(ing)?\s+of\b'
            ],
            'comparative': [
                r'\bdifference\s+between\b',
                r'\bcompare\b',
                r'\bvs\b|\bversus\b',
                r'\bbetter\s+than\b'
            ],
            'causal': [
                r'\bwhy\s+does\b',
                r'\bcause\s+of\b',
                r'\breason\s+for\b',
                r'\bbecause\s+of\b'
            ]
        }
    
    def _load_concept_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Load concept dictionary with related terms"""
        return {
            'machine learning': {
                'type': 'technical',
                'related': ['AI', 'artificial intelligence', 'neural networks', 'deep learning', 'algorithms']
            },
            'data science': {
                'type': 'technical', 
                'related': ['analytics', 'statistics', 'visualization', 'big data', 'modeling']
            },
            'web development': {
                'type': 'technical',
                'related': ['frontend', 'backend', 'HTML', 'CSS', 'JavaScript', 'frameworks']
            },
            'project management': {
                'type': 'business',
                'related': ['planning', 'scheduling', 'resources', 'timeline', 'deliverables']
            },
            'marketing': {
                'type': 'business',
                'related': ['advertising', 'promotion', 'branding', 'campaigns', 'customers']
            }
        }
    
    def _get_morphological_variations(self, word: str) -> List[str]:
        """Get morphological variations of a word"""
        variations = []
        
        # Simple English morphology rules
        if word.endswith('s') and len(word) > 3:
            variations.append(word[:-1])  # Remove plural 's'
        
        if word.endswith('ed') and len(word) > 4:
            variations.append(word[:-2])  # Remove past tense 'ed'
        
        if word.endswith('ing') and len(word) > 5:
            variations.append(word[:-3])  # Remove gerund 'ing'
        
        if word.endswith('ly') and len(word) > 4:
            variations.append(word[:-2])  # Remove adverb 'ly'
        
        # Add common suffixes
        if not word.endswith('s'):
            variations.append(word + 's')
        
        if not word.endswith('ed') and not word.endswith('ing'):
            variations.extend([word + 'ed', word + 'ing'])
        
        return variations
    
    def _get_domain_expansions(self, keyword: str) -> List[str]:
        """Get domain-specific expansions for a keyword"""
        
        domain_expansions = {
            'code': ['programming', 'development', 'software', 'script'],
            'data': ['information', 'dataset', 'records', 'statistics'],
            'system': ['platform', 'infrastructure', 'framework', 'architecture'],
            'user': ['customer', 'client', 'person', 'individual'],
            'process': ['procedure', 'workflow', 'method', 'approach'],
            'problem': ['issue', 'challenge', 'difficulty', 'bug'],
            'solution': ['answer', 'fix', 'resolution', 'approach']
        }
        
        return domain_expansions.get(keyword.lower(), [])
    
    def _update_statistics(self, intent: QueryIntent, complexity: QueryComplexity, expansion_ratio: float):
        """Update processing statistics"""
        
        # Update intent distribution
        intent_key = intent.value
        self.processing_stats['intent_distribution'][intent_key] = (
            self.processing_stats['intent_distribution'].get(intent_key, 0) + 1
        )
        
        # Update complexity distribution
        complexity_key = complexity.value
        self.processing_stats['complexity_distribution'][complexity_key] = (
            self.processing_stats['complexity_distribution'].get(complexity_key, 0) + 1
        )
        
        # Update average expansion ratio
        current_avg = self.processing_stats['avg_expansion_ratio']
        total_queries = self.processing_stats['total_queries']
        self.processing_stats['avg_expansion_ratio'] = (
            (current_avg * (total_queries - 1) + expansion_ratio) / total_queries
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get query processing statistics"""
        return self.processing_stats.copy()
