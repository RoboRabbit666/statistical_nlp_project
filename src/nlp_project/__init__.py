"""
Advanced Natural Language Processing Pipeline

A comprehensive NLP package featuring:
- RAG (Retrieval-Augmented Generation) systems
- Keyword extraction with multiple approaches
- Advanced sentence ranking and similarity
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .rag import RAGSystem
from .keyword_extraction import KeywordExtractor
from .sentence_ranking import SentenceRanker

__all__ = ["RAGSystem", "KeywordExtractor", "SentenceRanker"]