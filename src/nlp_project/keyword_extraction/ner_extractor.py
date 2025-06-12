"""Named Entity Recognition based keyword extraction."""

from typing import List, Dict, Any, Optional
import spacy
from spacy.lang.en import English

from ..utils.logger import get_logger

logger = get_logger(__name__)


class NERExtractor:
    """
    Named Entity Recognition based keyword extractor.
    
    Uses spaCy's NER capabilities to extract named entities
    as keywords from text.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the NER extractor.
        
        Args:
            spacy_model: SpaCy model name to use
        """
        self.model_name = spacy_model
        self.nlp = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"Failed to load {self.model_name}, using basic English model")
            try:
                self.nlp = English()
                # Add basic components
                if "tok2vec" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("tok2vec")
                if "ner" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("ner")
            except Exception as e:
                logger.error(f"Failed to initialize basic English model: {e}")
                raise
                
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords using NER.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            return []
            
        try:
            doc = self.nlp(text)
            keywords = []
            
            for ent in doc.ents:
                if ent.text.strip() and len(ent.text.strip()) > 1:
                    keywords.append(ent.text.strip())
                    
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in keywords:
                if keyword.lower() not in seen:
                    seen.add(keyword.lower())
                    unique_keywords.append(keyword)
                    
            logger.debug(f"NER extracted {len(unique_keywords)} keywords")
            return unique_keywords
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []
    
    def extract_keywords_with_scores(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract keywords with confidence scores and entity types.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping keywords to metadata
        """
        if not text or not text.strip():
            return {}
            
        try:
            doc = self.nlp(text)
            keyword_data = {}
            
            for ent in doc.ents:
                if ent.text.strip() and len(ent.text.strip()) > 1:
                    keyword = ent.text.strip()
                    
                    # Use entity confidence if available, otherwise default
                    confidence = getattr(ent, 'score', 0.8)
                    
                    keyword_data[keyword] = {
                        'confidence': confidence,
                        'entity_type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'source': 'ner'
                    }
                    
            logger.debug(f"NER extracted {len(keyword_data)} keywords with scores")
            return keyword_data
            
        except Exception as e:
            logger.error(f"NER extraction with scores failed: {e}")
            return {}
    
    def get_entities_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Get entities grouped by their types.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not text or not text.strip():
            return {}
            
        try:
            doc = self.nlp(text)
            entities_by_type = {}
            
            for ent in doc.ents:
                if ent.text.strip() and len(ent.text.strip()) > 1:
                    entity_type = ent.label_
                    entity_text = ent.text.strip()
                    
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                        
                    if entity_text not in entities_by_type[entity_type]:
                        entities_by_type[entity_type].append(entity_text)
                        
            logger.debug(f"Found entities of {len(entities_by_type)} types")
            return entities_by_type
            
        except Exception as e:
            logger.error(f"Entity type extraction failed: {e}")
            return {}
    
    def filter_entities_by_type(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[str]:
        """
        Extract only entities of specified types.
        
        Args:
            text: Input text
            entity_types: List of entity types to include
            
        Returns:
            List of filtered entities
        """
        if not text or not entity_types:
            return []
            
        try:
            entities_by_type = self.get_entities_by_type(text)
            filtered_entities = []
            
            for entity_type in entity_types:
                if entity_type in entities_by_type:
                    filtered_entities.extend(entities_by_type[entity_type])
                    
            logger.debug(f"Filtered to {len(filtered_entities)} entities")
            return filtered_entities
            
        except Exception as e:
            logger.error(f"Entity filtering failed: {e}")
            return []