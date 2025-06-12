"""Text preprocessing utilities."""

import string
import re
from typing import List, Union
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Advanced text preprocessing utilities for NLP tasks."""
    
    def __init__(self, remove_punctuation: bool = True, lowercase: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert text to lowercase
        """
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        
    def preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess a single sentence.
        
        Args:
            sentence: Input sentence to preprocess
            
        Returns:
            Preprocessed sentence
        """
        if not isinstance(sentence, str):
            logger.warning(f"Input is not a string: {type(sentence)}")
            return str(sentence)
            
        # Convert to lowercase
        if self.lowercase:
            sentence = sentence.lower()
            
        # Remove punctuation
        if self.remove_punctuation:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            
        # Remove extra whitespace and newlines
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        sentence = sentence.replace('\n', '').replace('\t', '')
        
        return sentence
    
    def preprocess_batch(self, sentences: List[str]) -> List[str]:
        """
        Preprocess a batch of sentences.
        
        Args:
            sentences: List of sentences to preprocess
            
        Returns:
            List of preprocessed sentences
        """
        if not isinstance(sentences, list):
            raise TypeError("Input must be a list of strings")
            
        return [self.preprocess_sentence(sentence) for sentence in sentences]
    
    def clean_text(self, text: str) -> str:
        """
        Perform advanced text cleaning.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Simple sentence tokenization.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]