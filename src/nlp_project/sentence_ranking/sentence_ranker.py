"""Advanced sentence ranking using fine-tuned sentence transformers."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.preprocessing import TextPreprocessor

logger = get_logger(__name__)


class SentenceRanker:
    """
    Advanced sentence ranking system using fine-tuned sentence transformers.
    
    This class provides sophisticated sentence similarity computation and ranking
    capabilities using pre-trained and fine-tuned transformer models.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        fine_tuned_model_path: Optional[str] = None
    ):
        """
        Initialize the sentence ranker.
        
        Args:
            config: Configuration object
            model_name: Name of the pre-trained model
            fine_tuned_model_path: Path to fine-tuned model weights
        """
        self.config = config or Config()
        self.model_name = model_name or self.config.sentence_model
        self.fine_tuned_model_path = fine_tuned_model_path
        
        # Initialize preprocessing
        self.preprocessor = TextPreprocessor()
        
        # Device setup
        self.device = self._setup_device()
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        logger.info(f"SentenceRanker initialized with model: {self.model_name}")
    
    def _setup_device(self) -> torch.device:
        """Set up the appropriate device for computation."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
            
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self) -> None:
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Load fine-tuned weights if provided
            if self.fine_tuned_model_path and Path(self.fine_tuned_model_path).exists():
                logger.info(f"Loading fine-tuned weights from: {self.fine_tuned_model_path}")
                self.model.load_state_dict(
                    torch.load(self.fine_tuned_model_path, map_location=self.device)
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            text: Input text to encode
            
        Returns:
            Text embeddings
        """
        try:
            tokenized = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            with torch.no_grad():
                outputs = self.model(**tokenized)
                embeddings = outputs.pooler_output
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def _encode_batch(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Encode a batch of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of text embeddings
        """
        embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                tokenized = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length,
                    return_tensors="pt"
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                
                with torch.no_grad():
                    outputs = self.model(**tokenized)
                    batch_embeddings = outputs.pooler_output
                    
                embeddings.extend([emb.unsqueeze(0) for emb in batch_embeddings])
                
            except Exception as e:
                logger.error(f"Failed to encode batch: {e}")
                raise
                
        return embeddings
    
    def rank_sentences_by_relevance(
        self,
        claim: str,
        sentences: List[str],
        return_similarities: bool = False
    ) -> Tuple[List[str], Optional[List[float]]]:
        """
        Rank sentences by relevance to a claim.
        
        Args:
            claim: The claim to compare against
            sentences: List of sentences to rank
            return_similarities: Whether to return similarity scores
            
        Returns:
            Tuple of (ranked_sentences, similarities if requested)
        """
        if not sentences:
            logger.warning("Empty sentences list provided")
            return [], [] if return_similarities else None
            
        try:
            # Preprocess inputs
            preprocessed_claim = self.preprocessor.preprocess_sentence(claim)
            preprocessed_sentences = [
                self.preprocessor.preprocess_sentence(sent) for sent in sentences
            ]
            
            # Encode claim
            claim_embedding = self._encode_text(preprocessed_claim)
            
            # Encode sentences in batches
            sentence_embeddings = self._encode_batch(preprocessed_sentences)
            
            # Calculate similarities
            similarities = []
            for sentence_embedding in sentence_embeddings:
                similarity = F.cosine_similarity(claim_embedding, sentence_embedding).item()
                similarities.append(similarity)
            
            # Create similarity-index pairs and sort
            similarity_with_index = [
                (i, sim) for i, sim in enumerate(similarities)
            ]
            similarity_with_index.sort(key=lambda x: x[1], reverse=True)
            
            # Get ranked sentences and similarities
            ranked_sentences = [sentences[i] for i, _ in similarity_with_index]
            sorted_similarities = [sim for _, sim in similarity_with_index]
            
            logger.info(f"Ranked {len(sentences)} sentences by relevance")
            
            if return_similarities:
                return ranked_sentences, sorted_similarities
            return ranked_sentences, None
            
        except Exception as e:
            logger.error(f"Failed to rank sentences: {e}")
            raise
    
    def calculate_similarity_matrix(
        self,
        sentences: List[str]
    ) -> np.ndarray:
        """
        Calculate pairwise similarity matrix for sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix as numpy array
        """
        if not sentences:
            return np.array([])
            
        try:
            # Preprocess sentences
            preprocessed_sentences = [
                self.preprocessor.preprocess_sentence(sent) for sent in sentences
            ]
            
            # Encode all sentences
            embeddings = self._encode_batch(preprocessed_sentences)
            
            # Calculate pairwise similarities
            n = len(embeddings)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i, n):
                    similarity = F.cosine_similarity(embeddings[i], embeddings[j]).item()
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                    
            logger.info(f"Calculated similarity matrix for {n} sentences")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity matrix: {e}")
            raise
    
    def get_top_k_similar(
        self,
        query: str,
        candidates: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most similar sentences to a query.
        
        Args:
            query: Query sentence
            candidates: Candidate sentences
            k: Number of top results to return
            
        Returns:
            List of (sentence, similarity_score) tuples
        """
        if not candidates:
            return []
            
        try:
            ranked_sentences, similarities = self.rank_sentences_by_relevance(
                query, candidates, return_similarities=True
            )
            
            # Return top-k results
            top_k = min(k, len(ranked_sentences))
            return [
                (ranked_sentences[i], similarities[i]) 
                for i in range(top_k)
            ]
            
        except Exception as e:
            logger.error(f"Failed to get top-k similar sentences: {e}")
            raise