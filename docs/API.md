# API Reference

## RAG System (`nlp_project.rag`)

### RAGSystem

Main class for retrieval-augmented generation fact-checking.

```python
class RAGSystem:
    def __init__(self, config: Optional[Config] = None)
    
    def verify_claim(
        self,
        claim: str,
        num_sentences: Optional[int] = None,
        return_evidence: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]
    
    def batch_verify_claims(
        self,
        claims: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]
    
    def evaluate_performance(
        self,
        test_data: List[Dict[str, str]],
        verdict_key: str = "label",
        claim_key: str = "claim"
    ) -> Dict[str, float]
```

### WikipediaRetriever

Handles Wikipedia content retrieval.

```python
class WikipediaRetriever:
    def search_wikipedia(self, query: str, limit: int = 1) -> List[str]
    
    def get_page_content(self, title: str) -> Optional[str]
    
    def search_and_get_content(self, query: str) -> List[str]
    
    def extract_sentences_from_content(
        self,
        content: str,
        max_sentences: Optional[int] = None
    ) -> List[str]
```

### ClaudeGenerator

Generates reasoning and verdicts using Claude API.

```python
class ClaudeGenerator:
    def generate_reasoning(
        self,
        claim: str,
        evidence: List[str]
    ) -> str
    
    def generate_verdict(
        self,
        claim: str,
        evidence: List[str],
        reasoning: Optional[str] = None
    ) -> str
    
    def generate_explanation(
        self,
        claim: str,
        evidence: List[str],
        verdict: str,
        reasoning: Optional[str] = None
    ) -> str
```

## Keyword Extraction (`nlp_project.keyword_extraction`)

### KeywordExtractor

Main keyword extraction class combining multiple approaches.

```python
class KeywordExtractor:
    def __init__(
        self,
        config: Optional[Config] = None,
        use_ner: bool = True,
        use_llm: bool = True,
        spacy_model: str = "en_core_web_sm"
    )
    
    def extract_keywords(
        self,
        text: str,
        method: str = "combined",
        max_keywords: Optional[int] = None
    ) -> List[str]
    
    def extract_keywords_with_scores(
        self,
        text: str,
        method: str = "combined"
    ) -> Dict[str, Dict[str, Any]]
    
    def validate_keywords(
        self,
        keywords: List[str],
        text: str
    ) -> List[str]
```

### NERExtractor

Named Entity Recognition based extraction.

```python
class NERExtractor:
    def extract_keywords(self, text: str) -> List[str]
    
    def extract_keywords_with_scores(self, text: str) -> Dict[str, Dict[str, Any]]
    
    def get_entities_by_type(self, text: str) -> Dict[str, List[str]]
    
    def filter_entities_by_type(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[str]
```

### LLMExtractor

Large Language Model based extraction.

```python
class LLMExtractor:
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]
    
    def extract_keywords_with_scores(self, text: str) -> Dict[str, Dict[str, Any]]
    
    def extract_wikipedia_queries(self, text: str) -> List[str]
```

## Sentence Ranking (`nlp_project.sentence_ranking`)

### SentenceRanker

Advanced sentence similarity and ranking.

```python
class SentenceRanker:
    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        fine_tuned_model_path: Optional[str] = None
    )
    
    def rank_sentences_by_relevance(
        self,
        claim: str,
        sentences: List[str],
        return_similarities: bool = False
    ) -> Tuple[List[str], Optional[List[float]]]
    
    def calculate_similarity_matrix(
        self,
        sentences: List[str]
    ) -> np.ndarray
    
    def get_top_k_similar(
        self,
        query: str,
        candidates: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]
```

### SimilarityCalculator

Additional similarity metrics.

```python
class SimilarityCalculator:
    def cosine_similarity_batch(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor
    
    def calculate_ndcg(
        self,
        similarities: List[float],
        relevance_scores: List[float],
        k: Optional[int] = None
    ) -> float
    
    def calculate_precision_at_k(
        self,
        similarities: List[float],
        relevance_scores: List[float],
        k: int,
        threshold: float = 0.5
    ) -> float
```

## Configuration (`nlp_project.utils`)

### Config

Configuration management class.

```python
class Config:
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    anthropic_model: str = "claude-3-haiku-20240307"
    max_tokens: int = 4096
    temperature: float = 0.0
    batch_size: int = 128
    max_sequence_length: int = 512
    num_retrieved_sentences: int = 5
    device: str = "auto"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config"
    
    @classmethod
    def from_env(cls) -> "Config"
    
    def to_dict(self) -> Dict[str, Any]
```

### Utility Functions

```python
def get_logger(
    name: str,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger

class TextPreprocessor:
    def preprocess_sentence(self, sentence: str) -> str
    def preprocess_batch(self, sentences: List[str]) -> List[str]
    def clean_text(self, text: str) -> str
    def tokenize_sentences(self, text: str) -> List[str]
```

## Return Types

### Verification Result

```python
{
    "claim": str,
    "verdict": str,  # "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
    "confidence": float,  # 0.0 to 1.0
    "keywords": List[str],
    "reasoning": str,
    "evidence": List[str]  # if return_evidence=True
}
```

### Keyword Data

```python
{
    "keyword": {
        "confidence": float,
        "entity_type": str,
        "sources": List[str],
        "start": int,  # character position (NER only)
        "end": int     # character position (NER only)
    }
}
```

### Performance Metrics

```python
{
    "accuracy": float,
    "correct": int,
    "total": int,
    "verdict_distribution": {
        "SUPPORTS": int,
        "REFUTES": int,
        "NOT ENOUGH INFO": int
    }
}
```