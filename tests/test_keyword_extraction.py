"""Tests for keyword extraction components."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp_project.keyword_extraction import KeywordExtractor
from nlp_project.utils import Config


class TestKeywordExtractor:
    """Test cases for keyword extractor."""
    
    @pytest.fixture
    def config(self):
        return Config(anthropic_api_key="test-key")
    
    @pytest.fixture
    def extractor(self, config):
        return KeywordExtractor(config)
    
    def test_extract_keywords_empty_text(self, extractor):
        """Test keyword extraction with empty text."""
        result = extractor.extract_keywords("")
        assert result == []
    
    def test_extract_keywords_with_text(self, extractor):
        """Test keyword extraction with sample text."""
        text = "Natural language processing is important."
        with patch.object(extractor, '_extract_with_llm') as mock_llm:
            mock_llm.return_value = ["natural language processing"]
            result = extractor.extract_keywords(text)
            assert len(result) > 0