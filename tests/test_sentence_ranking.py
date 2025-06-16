"""Tests for sentence ranking components."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp_project.sentence_ranking import SentenceRanker
from nlp_project.utils import Config


class TestSentenceRanker:
    """Test cases for sentence ranker."""
    
    @pytest.fixture
    def config(self):
        return Config()
    
    @pytest.fixture
    def ranker(self, config):
        return SentenceRanker(config)
    
    def test_rank_sentences_empty_list(self, ranker):
        """Test sentence ranking with empty list."""
        result = ranker.rank_sentences("test query", [])
        assert result == []
    
    def test_rank_sentences_with_data(self, ranker):
        """Test sentence ranking with sample data."""
        query = "machine learning"
        sentences = ["ML is great", "Weather is nice"]
        result = ranker.rank_sentences(query, sentences)
        assert len(result) == len(sentences)