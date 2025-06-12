"""Claude-based text generation for RAG system."""

from typing import List, Optional, Dict, Any
import re
import anthropic

from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ClaudeGenerator:
    """
    Claude-based text generator for RAG applications.
    
    Handles reasoning generation and verdict classification
    using Anthropic's Claude API.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Claude generator.
        
        Args:
            config: Configuration object containing API settings
        """
        self.config = config or Config()
        
        # Initialize Anthropic client
        api_key = self.config.anthropic_api_key
        if not api_key:
            raise ValueError("Anthropic API key not provided in configuration")
            
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    def _get_completion(
        self,
        prompt: str,
        system_prompt: str = "You are a professional fact checker.",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get completion from Claude API.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens for response
            
        Returns:
            API response text
        """
        max_tokens = max_tokens or self.config.max_tokens
        
        try:
            message = self.client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API request failed: {e}")
            raise
    
    def generate_reasoning(
        self,
        claim: str,
        evidence: List[str]
    ) -> str:
        """
        Generate reasoning for claim verification.
        
        Args:
            claim: Claim to analyze
            evidence: List of evidence sentences
            
        Returns:
            Generated reasoning text
        """
        if not claim or not evidence:
            logger.warning("Empty claim or evidence provided for reasoning")
            return "Insufficient information to provide reasoning."
        
        try:
            evidence_text = "\\n".join(evidence)
            
            prompt = f"""
            Use the INFORMATION provided to analyze the CLAIM.
            Reason through the verification process step by step, writing your reasoning clearly.
            
            CLAIM: {claim}
            
            INFORMATION: {evidence_text}
            
            Provide a detailed analysis of how the information relates to the claim.
            Consider what the evidence supports, refutes, or leaves unclear.
            """
            
            system_prompt = """You are a professional fact checker. Analyze claims objectively using only the provided information. Be thorough but concise in your reasoning."""
            
            reasoning = self._get_completion(
                prompt, 
                system_prompt=system_prompt,
                max_tokens=self.config.max_tokens
            )
            
            logger.debug("Generated reasoning for claim verification")
            return reasoning.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return f"Error generating reasoning: {str(e)}"
    
    def generate_verdict(
        self,
        claim: str,
        evidence: List[str],
        reasoning: Optional[str] = None
    ) -> str:
        """
        Generate final verdict for claim verification.
        
        Args:
            claim: Claim to verify
            evidence: List of evidence sentences
            reasoning: Previous reasoning (optional)
            
        Returns:
            Verdict: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"
        """
        if not claim or not evidence:
            logger.warning("Empty claim or evidence provided for verdict")
            return "NOT ENOUGH INFO"
        
        try:
            evidence_text = "\\n".join(evidence)
            reasoning_section = f"\\n\\nREASONING: {reasoning}" if reasoning else ""
            
            prompt = f"""
            Based on the given INFORMATION ONLY, decide whether the CLAIM is supported, refuted, or there is not enough information to verify it.
            
            Summarize your conclusion in one word: "SUPPORTS", "REFUTES" or "NOT ENOUGH INFO".
            
            Rules:
            - If the INFORMATION directly supports the claim, return: "SUPPORTS"
            - If the INFORMATION directly refutes the claim, return: "REFUTES"  
            - If the INFORMATION does not provide enough evidence to support or refute the claim, return: "NOT ENOUGH INFO"
            
            INFORMATION: {evidence_text}{reasoning_section}
            
            CLAIM: {claim}
            
            Verdict:
            """
            
            system_prompt = """You are a professional fact checker. Provide only the verdict word: SUPPORTS, REFUTES, or NOT ENOUGH INFO. Do not include explanations."""
            
            verdict = self._get_completion(
                prompt,
                system_prompt=system_prompt,
                max_tokens=50
            )
            
            # Clean and validate verdict
            verdict = verdict.strip().upper().replace('.', '')
            
            # Ensure valid verdict
            valid_verdicts = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
            if verdict not in valid_verdicts:
                # Try to extract valid verdict from response
                for valid_verdict in valid_verdicts:
                    if valid_verdict in verdict:
                        verdict = valid_verdict
                        break
                else:
                    logger.warning(f"Invalid verdict generated: {verdict}")
                    verdict = "NOT ENOUGH INFO"
            
            logger.debug(f"Generated verdict: {verdict}")
            return verdict
            
        except Exception as e:
            logger.error(f"Failed to generate verdict: {e}")
            return "NOT ENOUGH INFO"
    
    def generate_explanation(
        self,
        claim: str,
        evidence: List[str],
        verdict: str,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Generate human-readable explanation for the verdict.
        
        Args:
            claim: Original claim
            evidence: Evidence sentences
            verdict: Generated verdict
            reasoning: Generated reasoning
            
        Returns:
            Human-readable explanation
        """
        if not claim or not verdict:
            return "Unable to generate explanation due to missing information."
        
        try:
            evidence_text = "\\n".join(evidence) if evidence else "No evidence available"
            reasoning_section = f"\\n\\nReasoning: {reasoning}" if reasoning else ""
            
            prompt = f"""
            Provide a clear, concise explanation for why the claim received this verdict based on the evidence.
            
            CLAIM: {claim}
            VERDICT: {verdict}
            EVIDENCE: {evidence_text}{reasoning_section}
            
            Write a 2-3 sentence explanation that a general audience can understand.
            Focus on the key evidence that led to this conclusion.
            """
            
            system_prompt = """You are explaining fact-checking results to a general audience. Be clear, accurate, and concise."""
            
            explanation = self._get_completion(
                prompt,
                system_prompt=system_prompt,
                max_tokens=200
            )
            
            logger.debug("Generated explanation for verdict")
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def generate_summary(
        self,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate summary of multiple verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Summary text
        """
        if not results:
            return "No results to summarize."
        
        try:
            # Calculate statistics
            total = len(results)
            supports = sum(1 for r in results if r.get('verdict') == 'SUPPORTS')
            refutes = sum(1 for r in results if r.get('verdict') == 'REFUTES')
            not_enough = sum(1 for r in results if r.get('verdict') == 'NOT ENOUGH INFO')
            
            prompt = f"""
            Summarize the fact-checking results for {total} claims:
            
            Results:
            - {supports} claims SUPPORTED
            - {refutes} claims REFUTED  
            - {not_enough} claims had NOT ENOUGH INFO
            
            Provide a brief 2-3 sentence summary of these results.
            """
            
            system_prompt = """Provide a concise summary of fact-checking results."""
            
            summary = self._get_completion(
                prompt,
                system_prompt=system_prompt,
                max_tokens=150
            )
            
            logger.debug("Generated summary for verification results")
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Error generating summary: {str(e)}"