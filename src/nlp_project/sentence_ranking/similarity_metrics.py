"""Additional similarity metrics for sentence ranking."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SimilarityCalculator:
    """
    Advanced similarity calculation utilities.
    
    Provides various similarity metrics for sentence comparison
    beyond basic cosine similarity.
    """
    
    def __init__(self):
        """Initialize the similarity calculator."""
        logger.info("SimilarityCalculator initialized")
    
    def cosine_similarity_batch(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate cosine similarity between two batches of embeddings.
        
        Args:
            embeddings1: First batch of embeddings
            embeddings2: Second batch of embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity_matrix = torch.mm(embeddings1_norm, embeddings2_norm.t())
        
        return similarity_matrix
    
    def euclidean_distance(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Calculate Euclidean distance between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Euclidean distance
        """
        distance = torch.dist(embedding1, embedding2, p=2)
        return distance.item()
    
    def manhattan_distance(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Calculate Manhattan distance between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Manhattan distance
        """
        distance = torch.dist(embedding1, embedding2, p=1)
        return distance.item()
    
    def calculate_ndcg(
        self,
        similarities: List[float],
        relevance_scores: List[float],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            similarities: Predicted similarity scores
            relevance_scores: Ground truth relevance scores
            k: Number of top results to consider
            
        Returns:
            NDCG score
        """
        if not similarities or not relevance_scores:
            return 0.0
        
        if len(similarities) != len(relevance_scores):
            logger.warning("Similarity and relevance score lengths don't match")
            return 0.0
        
        # Sort by similarities and get corresponding relevance scores
        sorted_pairs = sorted(
            zip(similarities, relevance_scores),
            key=lambda x: x[0],
            reverse=True
        )
        
        if k:
            sorted_pairs = sorted_pairs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, (_, relevance) in enumerate(sorted_pairs):
            dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        if k:
            ideal_relevance = ideal_relevance[:k]
        
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance):
            idcg += relevance / np.log2(i + 2)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def calculate_mean_rank(
        self,
        similarities: List[float],
        target_index: int
    ) -> float:
        """
        Calculate mean rank of target item.
        
        Args:
            similarities: List of similarity scores
            target_index: Index of the target item
            
        Returns:
            Rank of target item (1-indexed)
        """
        if target_index >= len(similarities):
            return len(similarities)
        
        target_similarity = similarities[target_index]
        
        # Count how many items have higher similarity
        rank = 1
        for similarity in similarities:
            if similarity > target_similarity:
                rank += 1
        
        return float(rank)
    
    def calculate_precision_at_k(
        self,
        similarities: List[float],
        relevance_scores: List[float],
        k: int,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            similarities: Predicted similarity scores
            relevance_scores: Ground truth relevance scores
            k: Number of top results to consider
            threshold: Relevance threshold
            
        Returns:
            Precision@K score
        """
        if not similarities or not relevance_scores or k <= 0:
            return 0.0
        
        # Sort by similarities and get top-k
        sorted_pairs = sorted(
            zip(similarities, relevance_scores),
            key=lambda x: x[0],
            reverse=True
        )[:k]
        
        # Count relevant items in top-k
        relevant_count = sum(1 for _, relevance in sorted_pairs if relevance >= threshold)
        
        precision = relevant_count / min(k, len(sorted_pairs))
        return precision
    
    def calculate_recall_at_k(
        self,
        similarities: List[float],
        relevance_scores: List[float],
        k: int,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            similarities: Predicted similarity scores
            relevance_scores: Ground truth relevance scores
            k: Number of top results to consider
            threshold: Relevance threshold
            
        Returns:
            Recall@K score
        """
        if not similarities or not relevance_scores or k <= 0:
            return 0.0
        
        # Count total relevant items
        total_relevant = sum(1 for relevance in relevance_scores if relevance >= threshold)
        
        if total_relevant == 0:
            return 0.0
        
        # Sort by similarities and get top-k
        sorted_pairs = sorted(
            zip(similarities, relevance_scores),
            key=lambda x: x[0],
            reverse=True
        )[:k]
        
        # Count relevant items in top-k
        relevant_count = sum(1 for _, relevance in sorted_pairs if relevance >= threshold)
        
        recall = relevant_count / total_relevant
        return recall