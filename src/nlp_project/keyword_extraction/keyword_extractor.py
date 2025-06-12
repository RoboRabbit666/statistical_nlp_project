"""
Advanced keyword extraction combining multiple approaches.

This module provides a comprehensive keyword extraction system that combines
Named Entity Recognition (NER) and Large Language Model (LLM) approaches
for optimal performance.
"""

from typing import List, Set, Optional, Dict, Any
import spacy
from spacy.lang.en import English

from ..utils.config import Config
from ..utils.logger import get_logger
from .ner_extractor import NERExtractor
from .llm_extractor import LLMExtractor

logger = get_logger(__name__)


class KeywordExtractor:
    """
    Advanced keyword extraction system combining multiple approaches.
    
    This class integrates NER and LLM-based keyword extraction methods
    to achieve superior performance through ensemble techniques.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        use_ner: bool = True,
        use_llm: bool = True,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize the keyword extractor.
        
        Args:
            config: Configuration object
            use_ner: Whether to use NER extraction
            use_llm: Whether to use LLM extraction
            spacy_model: SpaCy model to use for NER
        """
        self.config = config or Config()
        self.use_ner = use_ner
        self.use_llm = use_llm
        
        # Initialize extractors
        self.extractors = {}
        
        if self.use_ner:
            try:
                self.extractors['ner'] = NERExtractor(spacy_model=spacy_model)
                logger.info("NER extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NER extractor: {e}")
                self.use_ner = False
        
        if self.use_llm:
            try:
                self.extractors['llm'] = LLMExtractor(config=self.config)
                logger.info("LLM extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM extractor: {e}")
                self.use_llm = False
        
        if not self.extractors:
            raise RuntimeError("No keyword extractors could be initialized")
            
        logger.info(f"KeywordExtractor initialized with {len(self.extractors)} extractors")
    
    def extract_keywords(
        self,
        text: str,
        method: str = "combined",
        max_keywords: Optional[int] = None
    ) -> List[str]:
        """
        Extract keywords from text using specified method.
        
        Args:
            text: Input text for keyword extraction
            method: Extraction method ('ner', 'llm', 'combined')
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for keyword extraction")
            return []
        
        try:
            keywords = set()
            
            if method == "combined":
                # Use all available extractors
                for extractor_name, extractor in self.extractors.items():
                    try:
                        extracted = extractor.extract_keywords(text)
                        keywords.update(extracted)
                        logger.debug(f"{extractor_name} extracted: {extracted}")
                    except Exception as e:
                        logger.warning(f"Extractor {extractor_name} failed: {e}")
                        continue
                        
            elif method == "ner" and "ner" in self.extractors:
                keywords.update(self.extractors["ner"].extract_keywords(text))
                
            elif method == "llm" and "llm" in self.extractors:
                keywords.update(self.extractors["llm"].extract_keywords(text))
                
            else:
                logger.error(f"Invalid method '{method}' or extractor not available")
                raise ValueError(f"Method '{method}' not available")
            
            # Convert to list and limit results
            result = list(keywords)
            
            if max_keywords and len(result) > max_keywords:
                result = result[:max_keywords]
                
            logger.info(f"Extracted {len(result)} keywords using method '{method}'")
            return result
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise
    
    def extract_keywords_with_scores(
        self,
        text: str,
        method: str = "combined"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract keywords with confidence scores and metadata.
        
        Args:
            text: Input text for keyword extraction
            method: Extraction method ('ner', 'llm', 'combined')
            
        Returns:
            Dictionary mapping keywords to metadata
        """
        if not text or not text.strip():
            return {}
        
        try:
            keyword_data = {}
            
            if method == "combined":
                # Collect from all extractors
                for extractor_name, extractor in self.extractors.items():
                    try:
                        if hasattr(extractor, 'extract_keywords_with_scores'):
                            extracted_data = extractor.extract_keywords_with_scores(text)
                        else:
                            # Fallback for extractors without scoring
                            keywords = extractor.extract_keywords(text)
                            extracted_data = {
                                kw: {
                                    'confidence': 1.0,
                                    'source': extractor_name,
                                    'entity_type': 'UNKNOWN'
                                } for kw in keywords
                            }
                        
                        # Merge results
                        for keyword, data in extracted_data.items():
                            if keyword in keyword_data:
                                # Update existing entry
                                keyword_data[keyword]['sources'] = keyword_data[keyword].get('sources', [])
                                keyword_data[keyword]['sources'].append(extractor_name)
                                keyword_data[keyword]['confidence'] = max(
                                    keyword_data[keyword]['confidence'],
                                    data.get('confidence', 0.5)
                                )
                            else:
                                keyword_data[keyword] = {
                                    'confidence': data.get('confidence', 0.5),
                                    'sources': [extractor_name],
                                    'entity_type': data.get('entity_type', 'UNKNOWN')
                                }
                                
                    except Exception as e:
                        logger.warning(f"Extractor {extractor_name} failed: {e}")
                        continue
                        
            else:
                # Single extractor
                if method in self.extractors:
                    extractor = self.extractors[method]
                    if hasattr(extractor, 'extract_keywords_with_scores'):
                        keyword_data = extractor.extract_keywords_with_scores(text)
                    else:
                        keywords = extractor.extract_keywords(text)
                        keyword_data = {
                            kw: {
                                'confidence': 1.0,
                                'sources': [method],
                                'entity_type': 'UNKNOWN'
                            } for kw in keywords
                        }
                else:
                    raise ValueError(f"Method '{method}' not available")
            
            logger.info(f"Extracted {len(keyword_data)} keywords with scores")
            return keyword_data
            
        except Exception as e:
            logger.error(f"Keyword extraction with scores failed: {e}")
            raise
    
    def validate_keywords(
        self,
        keywords: List[str],
        text: str
    ) -> List[str]:
        """
        Validate extracted keywords against the original text.
        
        Args:
            keywords: List of keywords to validate
            text: Original text
            
        Returns:
            List of validated keywords
        """
        if not keywords or not text:
            return []
        
        text_lower = text.lower()
        validated = []
        
        for keyword in keywords:
            if keyword and keyword.lower() in text_lower:
                validated.append(keyword)
            else:
                logger.debug(f"Keyword '{keyword}' not found in text")
        
        logger.info(f"Validated {len(validated)}/{len(keywords)} keywords")
        return validated
    
    def get_available_methods(self) -> List[str]:
        """
        Get list of available extraction methods.
        
        Returns:
            List of available method names
        """
        methods = list(self.extractors.keys())
        if len(methods) > 1:
            methods.append("combined")
        return methods