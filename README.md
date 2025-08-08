# Advanced Natural Language Processing Project

A comprehensive NLP pipeline featuring **Retrieval-Augmented Generation (RAG)**, **keyword extraction**, and **sentence ranking** for fact-checking and claim verification.

![NLP Portfolio Banner](images/Image.png)

## Features

### **RAG-Enhanced Fact Checking**
- Wikipedia-based knowledge retrieval system
- Sentence similarity ranking using pre-trained transformers
- Claude-powered reasoning and verdict generation
- Comprehensive claim verification pipeline

### **Multi-Method Keyword Extraction**
- **Named Entity Recognition (NER)** using spaCy
- **Large Language Model (LLM)** extraction via Claude
- **Ensemble approach** combining both methods (87.7% accuracy)
- Intelligent keyword validation and scoring

### **Advanced Sentence Ranking**
- Pre-trained sentence transformers (all-MiniLM-L6-v2)
- Cosine similarity computation
- Batch processing optimization
- GPU acceleration support

### **Production-Ready Architecture**
- Modular, extensible design
- Comprehensive error handling
- Professional logging system
- Configuration management
- Type hints and documentation

## Research Overview

This project presents a comprehensive **RAG-enhanced NLP system** for automated claim verification and fact-checking. Our research combines retrieval-augmented generation with advanced keyword extraction and sentence ranking techniques to create a robust fact-checking pipeline.

**Key Research Findings:**
1. **RAG Enhancement**: Demonstrated 5% accuracy improvement (58% → 63%) using RAG over LLM-only approach
2. **Ensemble Keyword Extraction**: Combined NER and LLM approach achieving 87.7% accuracy
3. **Sentence Ranking**: Leveraged pre-trained all-MiniLM-L6-v2 model with cosine similarity
4. **Modular Pipeline**: Four-step architecture for comprehensive claim verification

**For Detailed Analysis**: See the [complete technical report](docs/project_report/technical_report.pdf) for in-depth methodology, experimental results, and comprehensive evaluation.

## Methodology

Our approach integrates three core methodological components based on the FEVER (Fact Extraction and VERification) dataset research:

### **1. Retrieval-Augmented Generation (RAG)**
- **Knowledge Base**: Wikipedia article corpus for comprehensive domain coverage
- **Retrieval Strategy**: Keyword-based search with semantic similarity ranking
- **Generation Model**: Claude-3-Haiku for reasoning and verdict generation
- **Pipeline Integration**: End-to-end automated fact-checking workflow

### **2. Multi-Method Keyword Extraction**
- **Named Entity Recognition**: spaCy-based entity extraction for structured information
- **Large Language Model Extraction**: Claude-powered semantic keyword identification
- **Ensemble Approach**: Weighted combination of both methods for optimal performance
- **Contextual Validation**: Intelligent scoring and filtering mechanisms

### **3. Sentence Ranking and Similarity**
- **Base Model**: all-MiniLM-L6-v2 (compressed BERT variant)
- **Similarity Metrics**: Cosine similarity between claim and sentence embeddings
- **Ranking Strategy**: Top-k most relevant sentences per Wikipedia page
- **Optimization**: Batch processing with GPU acceleration support

## Experimental Results

### **Performance Benchmarks**
Based on research findings from the technical report:

| Component | Method | Accuracy | Improvement |
|-----------|--------|----------|-------------|
| **Keyword Extraction** | LLM Only | 71.2% | Baseline |
| **Keyword Extraction** | NER Only | 77.6% | **+6.4%** |
| **Keyword Extraction** | **Ensemble** | **87.7%** | **+16.5%** |
| **Claim Verification** | LLM Only | 58% | Baseline |
| **Claim Verification** | **RAG Enhanced** | **63%** | **+5%** |

### **Research Results Summary**

**Keyword Extraction Performance**
- **Evaluation Method**: Comparison against FEVER dataset ground truth Wikipedia pages
- **Test Samples**: 1000 samples from FEVER test dataset for keyword extraction evaluation
- **NER Method**: 77.6% accuracy using spaCy named entity recognition
- **LLM Method**: 71.2% accuracy using Claude-3-Haiku keyword extraction
- **Combined Ensemble**: 87.7% accuracy aggregating both NER and LLM results

**RAG vs LLM-Only Comparison**
- **Test Dataset**: 282 samples from FEVER test split
- **Limitation**: Sample size constrained by Claude-3-Haiku rate limits (50,000 tokens/min)
- **LLM-Only Accuracy**: 58% (baseline approach)
- **RAG-Enhanced Accuracy**: 63% (5% improvement over baseline)
- **Classification Categories**: SUPPORTS/REFUTES/NOT ENOUGH INFO
- **Key Finding**: RAG approach reduced hallucinations and overconfident "REFUTES" classifications

**Research Limitations**
- Constrained to Claude-3-Haiku (smallest tier) due to cost and time limitations
- FEVER dataset from 2018 may have outdated Wikipedia ground truth
- Rate limiting affected sample size for comprehensive evaluation

## System Architecture & Pipeline

### **Complete Workflow Overview**

The system follows a **5-step pipeline** for claim verification (based on `rag_system.py` implementation):

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Claim   │ -> │ Step 1: Keyword │ -> │ Step 2: Content │
│                 │    │ Extraction      │    │ Retrieval       │
│                 │    │ (NER + LLM)     │    │ (Wikipedia API) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │                       │
                                 v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Step 5: Final   │ <- │ Step 4: Claude  │ <- │ Step 3: Sentence│
│ Confidence &    │    │ Reasoning &     │    │ Ranking &       │
│ Result Assembly │    │ Verdict Gen.    │    │ Evidence Filter │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Implementation Details**

#### **Step 1: Keyword Extraction** (`rag_system.py:84-87`)
```python
# Multi-method keyword extraction using combined approach
keywords = self.keyword_extractor.extract_keywords(
    claim, method="combined"  # Combines NER (spaCy) + LLM (Claude)
)
```

#### **Step 2: Wikipedia Content Retrieval** (`rag_system.py:92-101`)
```python
# Retrieve Wikipedia content for each extracted keyword
all_content = []
for keyword in keywords:
    try:
        content = self.retriever.search_and_get_content(keyword)
        if content:
            all_content.extend(content)
    except Exception as e:
        logger.warning(f"Failed to retrieve content for '{keyword}': {e}")
```

#### **Step 3: Sentence Extraction & Ranking** (`rag_system.py:113-127`)
```python
# Extract sentences from content and rank by relevance
retrieved_sentences = []
for content in all_content:
    sentences = self.retriever.extract_sentences_from_content(content, num_sentences)
    ranked_sentences = self.sentence_ranker.rank_sentences_by_relevance(
        claim, sentences
    )[0]  # Get ranked sentences only
    retrieved_sentences.extend(ranked_sentences[:num_sentences])
```

#### **Step 4: Claude Reasoning & Verdict Generation** (`rag_system.py:139-141`)
```python
# Two-step Claude generation process
reasoning = self.generator.generate_reasoning(claim, indexed_sentences)
verdict = self.generator.generate_verdict(claim, indexed_sentences, reasoning)
```

#### **Step 5: Confidence Calculation & Result Assembly** (`rag_system.py:143-154`)
```python
# Calculate confidence score and assemble final result
confidence = self._calculate_confidence(claim, keywords, retrieved_sentences, verdict)
result = {
    "claim": claim, "verdict": verdict, "confidence": confidence,
    "keywords": keywords, "reasoning": reasoning
}
```

### **Core Components**

#### **1. RAG System** (`nlp_project.rag.RAGSystem`)
- **Main Pipeline**: Orchestrates the complete fact-checking workflow
- **Batch Processing**: Supports multiple claims verification
- **Error Handling**: Comprehensive exception management
- **Confidence Scoring**: Heuristic-based confidence calculation

#### **2. Keyword Extractor** (`nlp_project.keyword_extraction.KeywordExtractor`)
- **NER Integration**: spaCy-based named entity recognition
- **LLM Integration**: Claude-based semantic keyword extraction
- **Ensemble Logic**: Combines multiple extraction methods
- **Validation**: Keyword relevance scoring and filtering

#### **3. Sentence Ranker** (`nlp_project.sentence_ranking.SentenceRanker`)
- **Model**: Pre-trained all-MiniLM-L6-v2 transformer
- **Similarity**: Cosine similarity between embeddings
- **Optimization**: Batch processing with GPU support
- **Flexibility**: Support for custom fine-tuned models

#### **4. Wikipedia Retriever** (`nlp_project.rag.WikipediaRetriever`)
- **Search**: Keyword-based Wikipedia content retrieval
- **Processing**: Sentence extraction and content filtering
- **Caching**: Efficient content management

#### **5. Claude Generator** (`nlp_project.rag.ClaudeGenerator`)
- **Reasoning**: Chain-of-thought reasoning generation
- **Verdict**: Final claim classification (SUPPORTS/REFUTES/NOT ENOUGH INFO)
- **Integration**: Seamless RAG pipeline integration

## Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Anthropic API key

### **Installation Steps**

```bash
# Clone the repository
git clone https://github.com/RoboRabbit666/statistical_nlp_project.git
cd statistical-nlp-project

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Download spaCy model
python -m spacy download en_core_web_sm
```

### **Environment Configuration**

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Optional: Set custom device
export DEVICE="cuda"  # or "cpu"
```

### **Verification**

```bash
# Test the installation
python -c "from nlp_project import RAGSystem; print('Installation successful!')"
```

## Usage

### **Basic Usage**

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

### **Advanced Usage Examples**

#### **Batch Processing**
```python
# Process multiple claims efficiently
claims = ["Claim 1", "Claim 2", ...]
results = rag_system.batch_verify_claims(
    claims, 
    show_progress=True
)
```

#### **Component-Level Usage**
```python
# Keyword extraction only
extractor = KeywordExtractor()
keywords = extractor.extract_keywords("Your text here", method="combined")

# Sentence ranking only
ranker = SentenceRanker()
ranked_sentences = ranker.rank_sentences_by_relevance(query, candidates)
```

#### **Performance Monitoring**
```python
# Get system information
info = rag_system.get_system_info()
print(f"Device: {info['model_info']['device']}")
print(f"Models: {info['model_info']}")
```

## Testing & Evaluation

### **Running Tests**

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

### **Evaluation Script**

```bash
# Run comprehensive evaluation
python scripts/run_evaluation.py --test-data path/to/test_data.json --output-dir results/

# Evaluate specific components
python scripts/run_evaluation.py --test-data path/to/test_data.json --components rag keyword
```

## Configuration

### **Configuration Files**

```yaml
# config/default.yaml
sentence_model: "sentence-transformers/all-MiniLM-L6-v2"
anthropic_model: "claude-3-haiku-20240307"
batch_size: 128
num_retrieved_sentences: 5
max_sequence_length: 512
temperature: 0.0
max_tokens: 4096
device: "auto"
```

### **Programmatic Configuration**

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

## Project Structure

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
│   │   ├── sentence_ranker.py
│   │   └── similarity_metrics.py
│   └── utils/                 # Utilities
│       ├── config.py
│       ├── logger.py
│       └── preprocessing.py
├── tests/                     # Test suite
│   ├── test_keyword_extraction.py 
│   ├── test_rag.py                 
│   └── test_sentence_ranking.py
├── docs/                      # Documentation
│   └── project_report/
│       └── technical_report.pdf
├── config/                    # Configuration files
│   └── default.yaml
├── scripts/                   # Utility scripts
│   └── run_evaluation.py
├── notebooks/                 # Jupyter notebooks
│   └── demo.ipynb
└── README.md
```

## Demo & Examples

Check out the interactive demo in `notebooks/demo.ipynb` which showcases:
- Multi-method keyword extraction
- Sentence ranking and similarity
- Complete RAG-based fact checking
- System configuration and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for UCL Statistical NLP course
- Uses state-of-the-art transformer models
- Powered by Anthropic's Claude API
- Utilizes Wikipedia's knowledge base
- Based on FEVER dataset research

## Links

Project Link: [https://github.com/RoboRabbit666/statistical_nlp_project](https://github.com/RoboRabbit666/statistical_nlp_project)