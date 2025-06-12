"""Tests for RAG system components."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp_project.rag import RAGSystem, WikipediaRetriever, ClaudeGenerator
from nlp_project.utils import Config


class TestWikipediaRetriever:
    """Test cases for Wikipedia retriever."""
    
    def test_search_wikipedia(self):
        """Test Wikipedia search functionality."""
        retriever = WikipediaRetriever()
        
        # Test with empty query
        results = retriever.search_wikipedia("")
        assert results == []
        
        # Test with valid query (mocked)
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'query': {
                    'search': [{'title': 'Test Page'}]
                }
            }
            mock_get.return_value = mock_response
            
            results = retriever.search_wikipedia("test query")
            assert results == ['Test Page']
    
    def test_extract_sentences_from_content(self):
        """Test sentence extraction from content."""
        retriever = WikipediaRetriever()
        
        # Test with empty content
        sentences = retriever.extract_sentences_from_content("")
        assert sentences == []
        
        # Test with valid content
        content = "This is a test sentence. This is another sentence. Short."
        sentences = retriever.extract_sentences_from_content(content)
        assert len(sentences) >= 2


class TestClaudeGenerator:
    """Test cases for Claude generator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(anthropic_api_key="test-key")
    
    @pytest.fixture
    def generator(self, config):
        """Create generator with mocked client."""
        with patch('anthropic.Anthropic'):
            return ClaudeGenerator(config)
    
    def test_generate_verdict(self, generator):
        """Test verdict generation."""
        claim = "Test claim"
        evidence = ["Evidence 1", "Evidence 2"]
        
        with patch.object(generator, '_get_completion') as mock_completion:
            mock_completion.return_value = "SUPPORTS"
            
            verdict = generator.generate_verdict(claim, evidence)
            assert verdict in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    
    def test_generate_reasoning(self, generator):
        """Test reasoning generation."""
        claim = "Test claim"
        evidence = ["Evidence 1", "Evidence 2"]
        
        with patch.object(generator, '_get_completion') as mock_completion:
            mock_completion.return_value = "This is test reasoning."
            
            reasoning = generator.generate_reasoning(claim, evidence)
            assert isinstance(reasoning, str)
            assert len(reasoning) > 0


class TestRAGSystem:
    """Test cases for RAG system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            anthropic_api_key="test-key",
            num_retrieved_sentences=3
        )
    
    def test_rag_initialization(self, config):
        """Test RAG system initialization."""
        with patch('nlp_project.rag.rag_system.KeywordExtractor'), \
             patch('nlp_project.rag.rag_system.WikipediaRetriever'), \
             patch('nlp_project.rag.rag_system.SentenceRanker'), \
             patch('nlp_project.rag.rag_system.ClaudeGenerator'):
            
            rag_system = RAGSystem(config)
            assert rag_system.config == config
    
    def test_verify_claim_empty(self, config):
        """Test claim verification with empty claim."""
        with patch('nlp_project.rag.rag_system.KeywordExtractor'), \
             patch('nlp_project.rag.rag_system.WikipediaRetriever'), \
             patch('nlp_project.rag.rag_system.SentenceRanker'), \
             patch('nlp_project.rag.rag_system.ClaudeGenerator'):
            
            rag_system = RAGSystem(config)
            result = rag_system.verify_claim("")
            
            assert result["verdict"] == "NOT ENOUGH INFO"
            assert result["confidence"] == 0.0
    
    def test_get_system_info(self, config):
        """Test system information retrieval."""
        with patch('nlp_project.rag.rag_system.KeywordExtractor'), \
             patch('nlp_project.rag.rag_system.WikipediaRetriever'), \
             patch('nlp_project.rag.rag_system.SentenceRanker') as mock_ranker, \
             patch('nlp_project.rag.rag_system.ClaudeGenerator'):
            
            # Mock device
            mock_ranker.return_value.device.type = "cpu"
            
            rag_system = RAGSystem(config)
            info = rag_system.get_system_info()
            
            assert "model_info" in info
            assert "config" in info
            assert "components" in info