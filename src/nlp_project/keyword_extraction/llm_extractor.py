"""Large Language Model based keyword extraction."""

from typing import List, Dict, Any, Optional
import re
import anthropic

from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LLMExtractor:
    """
    Large Language Model based keyword extractor.
    
    Uses Anthropic's Claude API to extract keywords from text
    using natural language instructions.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the LLM extractor.
        
        Args:
            config: Configuration object containing API key and settings
        """
        self.config = config or Config()
        
        # Initialize Anthropic client
        api_key = self.config.anthropic_api_key
        if not api_key:
            raise ValueError("Anthropic API key not provided in configuration")
            
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("LLM extractor initialized with Anthropic client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    def _get_completion(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Get completion from Claude API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens for response
            
        Returns:
            API response text
        """
        try:
            message = self.client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                system="You are a professional keyword extraction specialist.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords using LLM.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            return []
            
        try:
            prompt = f"""
            Given the following text, extract the most important keywords or key phrases that would be useful for searching Wikipedia or other knowledge bases:

            TEXT: {text}

            Extract up to {max_keywords} keywords or key phrases that are:
            1. Most relevant to the main topic
            2. Likely to return informative search results
            3. Specific enough to be useful

            Return ONLY the keywords/phrases, one per line, without numbers or bullets.
            Do not include any explanations or additional text.
            """
            
            response = self._get_completion(prompt, max_tokens=256)
            
            # Parse the response
            keywords = []
            for line in response.strip().split('\n'):
                keyword = line.strip().strip('"-,.')
                if keyword and len(keyword) > 1:
                    keywords.append(keyword)
                    
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in keywords:
                if keyword.lower() not in seen:
                    seen.add(keyword.lower())
                    unique_keywords.append(keyword)
                    
            logger.debug(f"LLM extracted {len(unique_keywords)} keywords")
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {e}")
            return []
    
    def extract_keywords_with_scores(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract keywords with confidence scores.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping keywords to metadata
        """
        if not text or not text.strip():
            return {}
            
        try:
            prompt = f"""
            Given the following text, extract important keywords with confidence scores:

            TEXT: {text}

            For each keyword, provide:
            1. The keyword/phrase
            2. A confidence score (0.0-1.0) indicating how relevant it is
            3. The category/type of the keyword

            Format your response as:
            keyword1|confidence_score|category
            keyword2|confidence_score|category

            Example:
            Machine Learning|0.95|TECHNOLOGY
            Natural Language Processing|0.90|TECHNOLOGY
            """
            
            response = self._get_completion(prompt, max_tokens=512)
            
            # Parse the response
            keyword_data = {}
            for line in response.strip().split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        keyword = parts[0].strip()
                        try:
                            confidence = float(parts[1].strip())
                        except ValueError:
                            confidence = 0.7  # Default confidence
                        category = parts[2].strip()
                        
                        if keyword and len(keyword) > 1:
                            keyword_data[keyword] = {
                                'confidence': confidence,
                                'entity_type': category,
                                'source': 'llm'
                            }
                            
            logger.debug(f"LLM extracted {len(keyword_data)} keywords with scores")
            return keyword_data
            
        except Exception as e:
            logger.error(f"LLM keyword extraction with scores failed: {e}")
            # Fallback to simple extraction
            keywords = self.extract_keywords(text)
            return {
                kw: {
                    'confidence': 0.7,
                    'entity_type': 'GENERAL',
                    'source': 'llm'
                } for kw in keywords
            }
    
    def extract_wikipedia_queries(self, text: str) -> List[str]:
        """
        Extract optimized Wikipedia search queries from text.
        
        Args:
            text: Input text
            
        Returns:
            List of Wikipedia search queries
        """
        if not text or not text.strip():
            return []
            
        try:
            prompt = f"""
            Given the following claim or text, extract Wikipedia search queries that would help find relevant information:

            CLAIM: {text}

            Extract 1-3 specific Wikipedia page titles or search queries that would return the most informative pages about this claim.

            Return ONLY the search queries, one per line.
            Make them specific enough to find exact Wikipedia pages.
            Do not include explanations.
            """
            
            response = self._get_completion(prompt, max_tokens=256)
            
            # Parse queries
            queries = []
            for line in response.strip().split('\n'):
                query = line.strip().strip('"-,.')
                if query and len(query) > 1:
                    queries.append(query)
                    
            logger.debug(f"LLM extracted {len(queries)} Wikipedia queries")
            return queries
            
        except Exception as e:
            logger.error(f"Wikipedia query extraction failed: {e}")
            return []