# Advanced Natural Language Processing Project

A comprehensive NLP pipeline featuring **Retrieval-Augmented Generation (RAG)**, **multi-modal keyword extraction**, and **advanced sentence ranking** for fact-checking and claim verification.

![NLP Portfolio Banner](images/Gemini_Generated_Image.png)

## 🌟 Features

### 🔍 **Advanced RAG System**
- Wikipedia-based knowledge retrieval
- Fine-tuned sentence similarity ranking
- Claude-powered reasoning and verdict generation
- Comprehensive fact-checking pipeline

### 🎯 **Multi-Modal Keyword Extraction**
- **Named Entity Recognition (NER)** using spaCy
- **Large Language Model (LLM)** extraction via Claude
- **Ensemble approach** combining both methods (87.7% accuracy)
- Intelligent keyword validation and scoring

### 📊 **Sophisticated Sentence Ranking**
- Fine-tuned sentence transformers
- Cosine similarity computation
- Batch processing optimization
- GPU acceleration support

### 🏗️ **Production-Ready Architecture**
- Modular, extensible design
- Comprehensive error handling
- Professional logging system
- Configuration management
- Type hints and documentation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RoboRabbit666/statistical-nlp-project.git
cd statistical-nlp-project

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Setup

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Optional: Set custom device
export DEVICE="cuda"  # or "cpu"
```

### Basic Usage

```python
from nlp_project import RAGSystem, KeywordExtractor, SentenceRanker
from nlp_project.utils import Config

# Initialize with configuration
config = Config.from_env()
rag_system = RAGSystem(config=config)

# Verify a claim
result = rag_system.verify_claim(
    "The Earth is the third planet from the Sun.",
    return_evidence=True,
    verbose=True
)

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Reasoning: {result['reasoning']}")
```

## 📚 Core Components

### 1. RAG System (`nlp_project.rag`)

Complete fact-checking pipeline combining retrieval and generation:

```python
from nlp_project.rag import RAGSystem

rag = RAGSystem()

# Single claim verification
result = rag.verify_claim("Climate change is caused by human activities.")

# Batch processing
claims = ["Claim 1", "Claim 2", "Claim 3"]
results = rag.batch_verify_claims(claims)

# Performance evaluation
test_data = [{"claim": "...", "label": "SUPPORTS"}, ...]
metrics = rag.evaluate_performance(test_data)
```

### 2. Keyword Extraction (`nlp_project.keyword_extraction`)

Multi-modal keyword extraction with ensemble methods:

```python
from nlp_project.keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()

# Extract keywords using combined approach
keywords = extractor.extract_keywords(
    "Machine learning models require large datasets.",
    method="combined"
)

# Get keywords with confidence scores
keyword_data = extractor.extract_keywords_with_scores(text)
```

### 3. Sentence Ranking (`nlp_project.sentence_ranking`)

Advanced similarity computation and ranking:

```python
from nlp_project.sentence_ranking import SentenceRanker

ranker = SentenceRanker()

# Rank sentences by relevance to query
query = "What is artificial intelligence?"
candidates = ["AI is...", "Machine learning...", "Deep learning..."]

ranked_sentences, similarities = ranker.rank_sentences_by_relevance(
    query, candidates, return_similarities=True
)

# Get top-k most similar
top_similar = ranker.get_top_k_similar(query, candidates, k=3)
```

## 🔧 Configuration

### Configuration Files

```yaml
# config/default.yaml
sentence_model: "sentence-transformers/all-MiniLM-L6-v2"
anthropic_model: "claude-3-haiku-20240307"
batch_size: 128
num_retrieved_sentences: 5
device: "auto"
```

### Programmatic Configuration

```python
from nlp_project.utils import Config

# Load from YAML
config = Config.from_yaml("config/default.yaml")

# Load from environment
config = Config.from_env()

# Create custom configuration
config = Config(
    sentence_model="custom-model",
    batch_size=64,
    device="cuda"
)
```

## 📊 Performance Results

### Keyword Extraction Accuracy
- **NER Only**: 77.6%
- **LLM Only**: 71.2%  
- **Combined Approach**: **87.7%**

### RAG System Performance
- Evaluated on FEVER dataset
- Comprehensive fact-checking pipeline
- Support for SUPPORTS/REFUTES/NOT ENOUGH INFO classifications

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nlp_project --cov-report=html

# Run specific test modules
pytest tests/test_rag.py
pytest tests/test_keyword_extraction.py
pytest tests/test_sentence_ranking.py
```

## 📁 Project Structure

```
nlp_project/
├── src/nlp_project/           # Main package
│   ├── rag/                   # RAG system components
│   │   ├── rag_system.py      # Main RAG pipeline
│   │   ├── retriever.py       # Wikipedia retrieval
│   │   └── generator.py       # Claude generation
│   ├── keyword_extraction/    # Keyword extraction
│   │   ├── keyword_extractor.py
│   │   ├── ner_extractor.py
│   │   └── llm_extractor.py
│   ├── sentence_ranking/      # Sentence similarity
│   │   └── sentence_ranker.py
│   └── utils/                 # Utilities
│       ├── config.py
│       ├── logger.py
│       └── preprocessing.py
├── tests/                     # Test suite
├── docs/                      # Documentation
├── config/                    # Configuration files
├── scripts/                   # Utility scripts
└── notebooks/                 # Jupyter notebooks
```

## 🚀 Advanced Usage

### Custom Model Integration

```python
# Use custom fine-tuned model
ranker = SentenceRanker(
    model_name="custom-model",
    fine_tuned_model_path="path/to/weights.ckpt"
)
```

### Batch Processing

```python
# Process multiple claims efficiently
claims = ["Claim 1", "Claim 2", ...]
results = rag_system.batch_verify_claims(
    claims, 
    show_progress=True
)
```

### Performance Monitoring

```python
# Get system information
info = rag_system.get_system_info()
print(f"Device: {info['model_info']['device']}")
print(f"Models: {info['model_info']}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for UCL Statistical NLP course
- Uses state-of-the-art transformer models
- Powered by Anthropic's Claude API
- Utilizes Wikipedia's knowledge base

## 🔗 Link


Project Link: [https://github.com/RoboRabbit666/statistical-nlp-project](https://github.com/RoboRabbit666/statistical-nlp-project)

---

**🎓 Academic Project** | **🔬 Research-Grade** | **🏭 Production-Ready**