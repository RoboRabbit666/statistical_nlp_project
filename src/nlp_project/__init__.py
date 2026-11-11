"""
Claim verification on FEVER with retrieval-augmented generation (RAG).

The package has three parts:
- keyword extraction with spaCy NER and Claude
- sentence ranking with all-MiniLM-L6-v2
- the RAG pipeline that ties them together
"""

__version__ = "1.0.0"
__author__ = "RoboRabbit666"

from .rag import RAGSystem
from .keyword_extraction import KeywordExtractor
from .sentence_ranking import SentenceRanker

__all__ = ["RAGSystem", "KeywordExtractor", "SentenceRanker"]