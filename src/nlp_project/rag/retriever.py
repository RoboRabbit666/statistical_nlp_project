"""Wikipedia content retrieval for RAG system."""

import requests
import wikipedia
from typing import List, Optional, Dict, Any
import re
from urllib.parse import quote

from ..utils.logger import get_logger
from ..utils.preprocessing import TextPreprocessor

logger = get_logger(__name__)


class WikipediaRetriever:
    """
    Advanced Wikipedia content retriever.
    
    Handles Wikipedia API interactions, content extraction,
    and preprocessing for RAG applications.
    """
    
    def __init__(self):
        """Initialize the Wikipedia retriever."""
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.preprocessor = TextPreprocessor()
        logger.info("Wikipedia retriever initialized")
    
    def search_wikipedia(self, query: str, limit: int = 1) -> List[str]:
        """
        Search Wikipedia for page titles.
        
        Args:
            query: Search query
            limit: Number of results to return
            
        Returns:
            List of Wikipedia page titles
        """
        if not query or not query.strip():
            return []
        
        try:
            # Use Wikipedia API for search
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit,
            }
            
            response = requests.get(self.base_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                titles = [item['title'] for item in data['query']['search']]
                logger.debug(f"Found {len(titles)} Wikipedia pages for query: {query}")
                return titles
            else:
                logger.warning(f"No search results for query: {query}")
                return []
                
        except Exception as e:
            logger.error(f"Wikipedia search failed for query '{query}': {e}")
            return []
    
    def get_page_content(self, title: str) -> Optional[str]:
        """
        Get full content of a Wikipedia page.
        
        Args:
            title: Wikipedia page title
            
        Returns:
            Page content or None if failed
        """
        if not title:
            return None
        
        try:
            # Use Wikipedia API to get page content
            content_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'exsectionformat': 'wiki'
            }
            
            response = requests.get(self.base_url, params=content_params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                page_id = next(iter(pages))
                
                if 'extract' in pages[page_id]:
                    content = pages[page_id]['extract']
                    logger.debug(f"Retrieved content for page: {title}")
                    return content
                else:
                    logger.warning(f"No content found for page: {title}")
                    return None
            else:
                logger.warning(f"Invalid response for page: {title}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get content for page '{title}': {e}")
            return None
    
    def search_and_get_content(self, query: str) -> List[str]:
        """
        Search Wikipedia and get content from top results.
        
        Args:
            query: Search query
            
        Returns:
            List of page contents
        """
        if not query or not query.strip():
            return []
        
        try:
            # Search for page titles
            titles = self.search_wikipedia(query, limit=1)
            
            if not titles:
                logger.warning(f"No Wikipedia pages found for query: {query}")
                return []
            
            # Get content for each title
            contents = []
            for title in titles:
                content = self.get_page_content(title)
                if content:
                    contents.append(content)
            
            logger.debug(f"Retrieved {len(contents)} page contents for query: {query}")
            return contents
            
        except Exception as e:
            logger.error(f"Search and retrieval failed for query '{query}': {e}")
            return []
    
    def extract_sentences_from_content(
        self,
        content: str,
        max_sentences: Optional[int] = None
    ) -> List[str]:
        """
        Extract sentences from Wikipedia content.
        
        Args:
            content: Wikipedia page content
            max_sentences: Maximum number of sentences to return
            
        Returns:
            List of sentences
        """
        if not content:
            return []
        
        try:
            # Simple sentence splitting
            sentences = self.preprocessor.tokenize_sentences(content)
            
            # Filter out very short sentences and clean
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and not sentence.startswith('=='):
                    # Clean the sentence
                    cleaned = self.preprocessor.clean_text(sentence)
                    if cleaned:
                        filtered_sentences.append(cleaned)
            
            # Limit number of sentences if specified
            if max_sentences and len(filtered_sentences) > max_sentences:
                filtered_sentences = filtered_sentences[:max_sentences]
            
            logger.debug(f"Extracted {len(filtered_sentences)} sentences from content")
            return filtered_sentences
            
        except Exception as e:
            logger.error(f"Failed to extract sentences: {e}")
            return []
    
    def get_page_summary(self, title: str) -> Optional[str]:
        """
        Get summary/introduction of a Wikipedia page.
        
        Args:
            title: Wikipedia page title
            
        Returns:
            Page summary or None if failed
        """
        if not title:
            return None
        
        try:
            # Use Wikipedia API to get page summary
            summary_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'exsectionformat': 'wiki'
            }
            
            response = requests.get(self.base_url, params=summary_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                page_id = next(iter(pages))
                
                if 'extract' in pages[page_id]:
                    summary = pages[page_id]['extract']
                    logger.debug(f"Retrieved summary for page: {title}")
                    return summary
                    
            logger.warning(f"No summary found for page: {title}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get summary for page '{title}': {e}")
            return None
    
    def search_multiple_queries(self, queries: List[str]) -> List[str]:
        """
        Search Wikipedia with multiple queries and combine results.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of combined page contents
        """
        if not queries:
            return []
        
        all_contents = []
        
        for query in queries:
            try:
                contents = self.search_and_get_content(query)
                all_contents.extend(contents)
                
            except Exception as e:
                logger.warning(f"Failed to process query '{query}': {e}")
                continue
        
        logger.info(f"Retrieved {len(all_contents)} total contents from {len(queries)} queries")
        return all_contents
    
    def validate_page_exists(self, title: str) -> bool:
        """
        Check if a Wikipedia page exists.
        
        Args:
            title: Wikipedia page title
            
        Returns:
            True if page exists, False otherwise
        """
        if not title:
            return False
        
        try:
            search_results = self.search_wikipedia(title, limit=1)
            return len(search_results) > 0
            
        except Exception as e:
            logger.error(f"Failed to validate page '{title}': {e}")
            return False