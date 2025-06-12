"""
Advanced Retrieval-Augmented Generation (RAG) System.

This module implements a comprehensive RAG system for fact-checking and
claim verification using Wikipedia knowledge retrieval and Claude generation.
"""

from typing import List, Dict, Optional, Tuple, Any
import re

from ..utils.config import Config
from ..utils.logger import get_logger
from ..keyword_extraction import KeywordExtractor
from ..sentence_ranking import SentenceRanker
from .retriever import WikipediaRetriever
from .generator import ClaudeGenerator

logger = get_logger(__name__)


class RAGSystem:
    """
    Comprehensive RAG system for fact-checking and claim verification.
    
    Combines keyword extraction, Wikipedia retrieval, sentence ranking,
    and LLM generation for robust claim verification.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the RAG system.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize components
        try:
            self.keyword_extractor = KeywordExtractor(config=self.config)
            self.retriever = WikipediaRetriever()
            self.sentence_ranker = SentenceRanker(config=self.config)
            self.generator = ClaudeGenerator(config=self.config)
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def verify_claim(
        self,
        claim: str,
        num_sentences: Optional[int] = None,
        return_evidence: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Verify a claim using the full RAG pipeline.
        
        Args:
            claim: Claim to verify
            num_sentences: Number of evidence sentences to retrieve
            return_evidence: Whether to return retrieved evidence
            verbose: Whether to include detailed information
            
        Returns:
            Dictionary containing verification result and metadata
        """
        if not claim or not claim.strip():
            logger.warning("Empty claim provided")
            return {
                "claim": claim,
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "error": "Empty claim"
            }
        
        num_sentences = num_sentences or self.config.num_retrieved_sentences
        
        try:
            logger.info(f"Verifying claim: {claim[:100]}...")
            
            # Step 1: Extract keywords
            keywords = self.keyword_extractor.extract_keywords(
                claim, method="combined"
            )
            
            if verbose:
                logger.info(f"Extracted keywords: {keywords}")
            
            # Step 2: Retrieve Wikipedia content
            all_content = []
            for keyword in keywords:
                try:
                    content = self.retriever.search_and_get_content(keyword)
                    if content:
                        all_content.extend(content)
                except Exception as e:
                    logger.warning(f"Failed to retrieve content for '{keyword}': {e}")
                    continue
            
            if not all_content:
                logger.warning("No content retrieved from Wikipedia")
                return {
                    "claim": claim,
                    "verdict": "NOT ENOUGH INFO",
                    "confidence": 0.0,
                    "keywords": keywords,
                    "error": "No content retrieved"
                }
            
            # Step 3: Extract and rank relevant sentences
            retrieved_sentences = []
            for content in all_content:
                try:
                    sentences = self.retriever.extract_sentences_from_content(
                        content, num_sentences
                    )
                    ranked_sentences = self.sentence_ranker.rank_sentences_by_relevance(
                        claim, sentences
                    )[0]  # Get ranked sentences only
                    retrieved_sentences.extend(ranked_sentences[:num_sentences])
                    
                except Exception as e:
                    logger.warning(f"Failed to process content: {e}")
                    continue
            
            # Limit total sentences and add indices
            retrieved_sentences = retrieved_sentences[:num_sentences * 2]  # Some redundancy
            indexed_sentences = [
                f"{i+1}. {sentence[:1000]}" 
                for i, sentence in enumerate(retrieved_sentences)
            ]
            
            if verbose:
                logger.info(f"Retrieved {len(indexed_sentences)} evidence sentences")
            
            # Step 4: Generate reasoning and verdict
            reasoning = self.generator.generate_reasoning(claim, indexed_sentences)
            verdict = self.generator.generate_verdict(claim, indexed_sentences, reasoning)
            
            # Step 5: Calculate confidence (simple heuristic)
            confidence = self._calculate_confidence(
                claim, keywords, retrieved_sentences, verdict
            )
            
            result = {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "keywords": keywords,
                "reasoning": reasoning
            }
            
            if return_evidence:
                result["evidence"] = indexed_sentences
                
            if verbose:
                result["num_content_sources"] = len(all_content)
                result["num_evidence_sentences"] = len(indexed_sentences)
            
            logger.info(f"Claim verification completed. Verdict: {verdict}")
            return result
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {
                "claim": claim,
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def batch_verify_claims(
        self,
        claims: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Verify multiple claims in batch.
        
        Args:
            claims: List of claims to verify
            show_progress: Whether to show progress information
            
        Returns:
            List of verification results
        """
        if not claims:
            return []
        
        results = []
        total = len(claims)
        
        logger.info(f"Starting batch verification of {total} claims")
        
        for i, claim in enumerate(claims):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing claim {i+1}/{total}")
                
            try:
                result = self.verify_claim(claim)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to verify claim {i+1}: {e}")
                results.append({
                    "claim": claim,
                    "verdict": "NOT ENOUGH INFO",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        logger.info(f"Batch verification completed. {len(results)} results")
        return results
    
    def evaluate_performance(
        self,
        test_data: List[Dict[str, str]],
        verdict_key: str = "label",
        claim_key: str = "claim"
    ) -> Dict[str, float]:
        """
        Evaluate RAG system performance on test data.
        
        Args:
            test_data: List of test examples with claims and labels
            verdict_key: Key for ground truth verdict
            claim_key: Key for claim text
            
        Returns:
            Dictionary with performance metrics
        """
        if not test_data:
            return {}
        
        logger.info(f"Evaluating performance on {len(test_data)} examples")
        
        correct = 0
        total = 0
        verdict_counts = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
        
        for example in test_data:
            try:
                claim = example[claim_key]
                true_label = example[verdict_key]
                
                result = self.verify_claim(claim)
                predicted_label = result["verdict"]
                
                if predicted_label in verdict_counts:
                    verdict_counts[predicted_label] += 1
                
                if predicted_label == true_label:
                    correct += 1
                    
                total += 1
                
            except Exception as e:
                logger.error(f"Evaluation failed for example: {e}")
                total += 1  # Count as incorrect
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "verdict_distribution": verdict_counts
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.3f}")
        return metrics
    
    def _calculate_confidence(
        self,
        claim: str,
        keywords: List[str],
        evidence: List[str],
        verdict: str
    ) -> float:
        """
        Calculate confidence score for the verdict.
        
        Args:
            claim: Original claim
            keywords: Extracted keywords
            evidence: Retrieved evidence sentences
            verdict: Generated verdict
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Adjust based on evidence quantity
        if len(evidence) >= 3:
            confidence += 0.2
        elif len(evidence) >= 1:
            confidence += 0.1
        
        # Adjust based on keyword relevance
        if len(keywords) >= 2:
            confidence += 0.1
        
        # Adjust based on verdict certainty
        if verdict in ["SUPPORTS", "REFUTES"]:
            confidence += 0.1
        
        # Check for keyword presence in evidence
        claim_lower = claim.lower()
        evidence_text = " ".join(evidence).lower()
        
        keyword_matches = sum(
            1 for kw in keywords 
            if kw.lower() in evidence_text
        )
        
        if keyword_matches > 0:
            confidence += min(0.1, keyword_matches * 0.05)
        
        return min(1.0, confidence)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system configuration.
        
        Returns:
            Dictionary with system information
        """
        return {
            "model_info": {
                "sentence_model": self.config.sentence_model,
                "anthropic_model": self.config.anthropic_model,
                "device": self.sentence_ranker.device.type
            },
            "config": {
                "batch_size": self.config.batch_size,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "num_retrieved_sentences": self.config.num_retrieved_sentences
            },
            "components": {
                "keyword_extractor": type(self.keyword_extractor).__name__,
                "retriever": type(self.retriever).__name__,
                "sentence_ranker": type(self.sentence_ranker).__name__,
                "generator": type(self.generator).__name__
            }
        }