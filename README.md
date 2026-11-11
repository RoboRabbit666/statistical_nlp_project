# Retrieval-augmented claim verification on FEVER

![A magnifying glass held over an open book, with glowing blue network lines](images/Image.png)

This repository holds our project for the Statistical NLP course at UCL. We compared two ways of labelling claims from the FEVER dataset (Fact Extraction and VERification) with a large language model (LLM). In the first, the model only saw the claim and had to rely on its own knowledge. In the second, it was also given evidence retrieved from Wikipedia, which is the idea behind retrieval-augmented generation (RAG). With Claude 3 Haiku, accuracy on 282 FEVER test claims rose from 58% to 63% when retrieval was added (i.e. 16 more claims were labelled correctly). However, the evidence also led the model to label more true claims as NOT ENOUGH INFO.

The code is in `src/nlp_project`, and the full write-up is in the [technical report](docs/project_report/technical_report.pdf), which is also where the numbers below come from.

## How it works

We split the method into three steps. Firstly, keywords were extracted from the claim with named entity recognition (NER) and with Claude, the two lists were merged, and one Wikipedia page was retrieved for each keyword. Then, the sentences of each page were ranked by cosine similarity to the claim, and the five most similar sentences from each page were kept as evidence (e.g. two keywords gave ten sentences). Finally, Claude was asked to reason step by step about the claim and the evidence, and then to give a one-word label (SUPPORTS, REFUTES or NOT ENOUGH INFO) based on that reasoning. For both steps that involve Claude, we used Claude 3 Haiku, which was relatively new and cheap to run, with prompts only and no further training.

The similarity model was all-MiniLM-L6-v2, a compressed version of BERT. We fine-tuned it with a triplet loss on FEVER (i.e. each claim paired with its evidence sentence and a random other sentence), so that a claim ends up closer to its evidence than to unrelated text. Afterwards, we compared six fine-tuned versions with the original model and kept the one with the highest normalised discounted cumulative gain (NDCG), which had all six layers unfrozen and 50,000 training samples. Its NDCG was approximately 0.73, against 0.54 for the original model (figure 1 of the report).

In the package, `RAGSystem.verify_claim` runs the whole chain for one claim (see `src/nlp_project/rag/rag_system.py`). It uses spaCy's `en_core_web_sm` model for NER, the MediaWiki API to fetch live Wikipedia pages, Hugging Face `transformers` to load all-MiniLM-L6-v2, and the Anthropic API for the Claude calls.

## Results

The table below gives the number of claims labelled correctly in each condition, taken from the confusion matrices in figure 3 of the report. Besides accuracy, the macro-averaged F1 score rose from 0.53 to 0.63 (figures 7 and 8 of the report).

| FEVER label (claims) | LLM only | LLM with retrieval |
|---|---|---|
| SUPPORTS (88) | 68 | 63 |
| REFUTES (90) | 83 | 71 |
| NOT ENOUGH INFO (104) | 12 | 45 |
| All (282) | 163 (58%) | 179 (63%) |

It can be seen that the net gain comes entirely from the NOT ENOUGH INFO claims, since the other two labels both lost correct answers. Without evidence, Claude labelled 74 of these 104 claims as REFUTES, whereas with retrieval it did so 40 times. However, retrieval made Claude more cautious about true claims, and 19 SUPPORTS claims were labelled NOT ENOUGH INFO, against 1 without retrieval. In the report, we took the drop in false REFUTES labels as a sign of less hallucination, although hallucination itself was not measured directly.

For keyword extraction, a claim counted as a hit only when one of the pages retrieved from its keywords was the evidence page given by FEVER, even though another page could still be relevant. On 1,000 test claims, NER alone found the evidence page for 77.6% of them and Claude alone for 71.2%. Nevertheless, NER sometimes returned no keyword at all (e.g. for "License to Wed is a movie"), whereas Claude consistently returned one. Hence, we merged the two lists, which raised the hit rate to 87.7% (i.e. 10.1 percentage points above NER alone).

## Limitations

The gain from retrieval is modest, and the report gives one run of each set-up with no statistical test. We limited the test set to 282 claims because of the API rate limit at the time (50,000 tokens per minute), as well as time and cost. Time and cost were also the reason for using only the smallest Claude 3 model, and a larger model or another prompting method (e.g. majority voting) may give different results. Claude sometimes reasoned correctly but still gave the wrong label. Also, claims with similar reasoning did not always get the same label, which is likely because the labels had no clear criteria and every claim ran in a separate session. Finally, FEVER dates from 2018, so part of its Wikipedia ground truth is outdated, and a correct SUPPORTS answer can be scored as wrong where FEVER says NOT ENOUGH INFO.

## Setting up

The package needs Python 3.8 or later and an Anthropic API key. It also needs network access, because Wikipedia and Claude are called at run time and the sentence model is downloaded from Hugging Face on first use.

```bash
git clone https://github.com/RoboRabbit666/statistical_nlp_project.git
cd statistical_nlp_project
pip install -e .
python -m spacy download en_core_web_sm
export ANTHROPIC_API_KEY="your-api-key-here"
```

`pip install -e .` installs everything in `requirements.txt`, including pytest for the tests. The sentence ranker runs on a CUDA GPU when one is available and on the CPU otherwise.

## Checking a claim

```python
from nlp_project import RAGSystem
from nlp_project.utils import Config

config = Config.from_env()
rag = RAGSystem(config=config)

result = rag.verify_claim(
    "The Earth is the third planet from the Sun.",
    return_evidence=True,
)

print("Verdict:", result["verdict"])
print(f"Confidence: {result['confidence']:.2f}")
print("Reasoning:", result.get("reasoning"))
for sentence in result.get("evidence", []):
    print(sentence)
```

`Config.from_env()` reads the API key from the environment, together with `DEVICE` and `SENTENCE_MODEL` if they are set. This config then has to be passed to `RAGSystem`, as in the example above. The other settings and their defaults are fields of `Config` in `src/nlp_project/utils/config.py`. The confidence is a score from a simple rule in the code (see `_calculate_confidence` in `rag_system.py`), not a probability.

`KeywordExtractor` and `SentenceRanker` can also be used on their own. `SentenceRanker` needs no API key, while `KeywordExtractor` only uses NER unless it is given a config with a key (e.g. `KeywordExtractor(config=config)`). Their methods are listed in `docs/API.md`, and `notebooks/demo.ipynb` goes through each part step by step. Jupyter is not in `requirements.txt` and has to be installed separately.

## Running the tests

```bash
pytest
pytest --cov=nlp_project --cov-report=html
```

The second command also writes a coverage report to `htmlcov/`. None of the tests needs an API key, but the sentence ranking tests download the model on their first run.

## Repository layout

```
statistical_nlp_project/
  src/nlp_project/
    keyword_extraction/   NER and Claude keyword extractors
    rag/                  Wikipedia retriever, Claude generator and RAGSystem
    sentence_ranking/     MiniLM sentence ranker and similarity metrics
    utils/                Config, logging and text preprocessing
  tests/                  pytest suite
  scripts/run_evaluation.py
  notebooks/demo.ipynb
  config/default.yaml
  docs/API.md
  docs/project_report/technical_report.pdf
  images/Image.png
  setup.py, requirements.txt, LICENSE
```

## Licence

The code is released under the MIT License (see [LICENSE](LICENSE)).
