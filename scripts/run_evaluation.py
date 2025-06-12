#!/usr/bin/env python3
"""
Evaluation script for the Advanced NLP Project.

This script provides comprehensive evaluation capabilities for the RAG system,
keyword extraction, and sentence ranking components.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp_project import RAGSystem, KeywordExtractor, SentenceRanker
from nlp_project.utils import Config, get_logger

logger = get_logger(__name__)


def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test data from file."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")
    
    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            return json.load(f)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def evaluate_rag_system(
    config: Config,
    test_data: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """Evaluate RAG system performance."""
    logger.info("Starting RAG system evaluation")
    
    rag_system = RAGSystem(config=config)
    
    # Run evaluation
    metrics = rag_system.evaluate_performance(test_data)
    
    # Save detailed results
    results_file = output_dir / "rag_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"RAG evaluation completed. Accuracy: {metrics.get('accuracy', 0):.3f}")
    return metrics


def evaluate_keyword_extraction(
    config: Config,
    test_data: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """Evaluate keyword extraction performance."""
    logger.info("Starting keyword extraction evaluation")
    
    extractor = KeywordExtractor(config=config)
    
    results = {
        "ner_only": [],
        "llm_only": [],
        "combined": []
    }
    
    for example in test_data:
        claim = example.get('claim', '')
        true_keywords = example.get('keywords', [])
        
        # Test each method
        for method in results.keys():
            try:
                extracted = extractor.extract_keywords(claim, method=method)
                
                # Calculate simple accuracy (if ground truth available)
                if true_keywords:
                    accuracy = len(set(extracted) & set(true_keywords)) / len(set(true_keywords)) if true_keywords else 0
                else:
                    accuracy = None
                
                results[method].append({
                    "claim": claim,
                    "extracted": extracted,
                    "true_keywords": true_keywords,
                    "accuracy": accuracy
                })
                
            except Exception as e:
                logger.error(f"Keyword extraction failed for method {method}: {e}")
                results[method].append({
                    "claim": claim,
                    "extracted": [],
                    "true_keywords": true_keywords,
                    "accuracy": 0.0,
                    "error": str(e)
                })
    
    # Calculate overall metrics
    metrics = {}
    for method, method_results in results.items():
        accuracies = [r["accuracy"] for r in method_results if r["accuracy"] is not None]
        metrics[method] = {
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "num_processed": len(method_results),
            "num_errors": sum(1 for r in method_results if "error" in r)
        }
    
    # Save results
    results_file = output_dir / "keyword_extraction_results.json"
    with open(results_file, 'w') as f:
        json.dump({"metrics": metrics, "detailed_results": results}, f, indent=2)
    
    logger.info("Keyword extraction evaluation completed")
    return metrics


def evaluate_sentence_ranking(
    config: Config,
    test_data: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """Evaluate sentence ranking performance."""
    logger.info("Starting sentence ranking evaluation")
    
    ranker = SentenceRanker(config=config)
    
    results = []
    total_examples = 0
    successful_rankings = 0
    
    for example in test_data:
        claim = example.get('claim', '')
        sentences = example.get('sentences', [])
        
        if not claim or not sentences:
            continue
            
        total_examples += 1
        
        try:
            # Rank sentences
            ranked_sentences, similarities = ranker.rank_sentences_by_relevance(
                claim, sentences, return_similarities=True
            )
            
            results.append({
                "claim": claim,
                "original_sentences": sentences,
                "ranked_sentences": ranked_sentences,
                "similarities": similarities
            })
            
            successful_rankings += 1
            
        except Exception as e:
            logger.error(f"Sentence ranking failed: {e}")
            results.append({
                "claim": claim,
                "original_sentences": sentences,
                "error": str(e)
            })
    
    # Calculate metrics
    success_rate = successful_rankings / total_examples if total_examples > 0 else 0
    
    metrics = {
        "success_rate": success_rate,
        "total_examples": total_examples,
        "successful_rankings": successful_rankings
    }
    
    # Save results
    results_file = output_dir / "sentence_ranking_results.json"
    with open(results_file, 'w') as f:
        json.dump({"metrics": metrics, "detailed_results": results}, f, indent=2)
    
    logger.info(f"Sentence ranking evaluation completed. Success rate: {success_rate:.3f}")
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Advanced NLP Project components")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["rag", "keyword", "sentence"],
        default=["rag", "keyword", "sentence"],
        help="Components to evaluate"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # Load configuration
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}. Using default config.")
        config = Config.from_env()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load test data
    try:
        test_data = load_test_data(args.test_data)
        logger.info(f"Loaded {len(test_data)} test examples")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return 1
    
    # Run evaluations
    all_metrics = {}
    
    try:
        if "rag" in args.components:
            all_metrics["rag"] = evaluate_rag_system(config, test_data, output_dir)
        
        if "keyword" in args.components:
            all_metrics["keyword"] = evaluate_keyword_extraction(config, test_data, output_dir)
        
        if "sentence" in args.components:
            all_metrics["sentence"] = evaluate_sentence_ranking(config, test_data, output_dir)
        
        # Save summary
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for component, metrics in all_metrics.items():
            print(f"\n{component.upper()} Results:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
        
        print(f"\nDetailed results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())