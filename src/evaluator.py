"""
evaluator.py
------------
Evaluates the Graph RAG pipeline using the RAGAS framework (v0.1.x).

Metrics measured:
  - faithfulness       : Is the answer grounded in the retrieved context?
  - answer_relevancy   : How relevant is the answer to the question?
  - context_recall     : How much of the ground-truth is covered by context?
  - context_precision  : Signal-to-noise ratio of retrieved context chunks.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from datasets import Dataset

# ---------------------------------------------------------------------------
# Import RAGAS metrics defensively — API changed across 0.1.x patch versions
# ---------------------------------------------------------------------------
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from ragas import evaluate as ragas_evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default evaluation dataset
# ---------------------------------------------------------------------------
DEFAULT_EVAL_DATASET = [
    {
        "question": "Who developed BERT and on which datasets was it trained?",
        "ground_truth": (
            "BERT was developed by Google AI in 2018. It was trained on the "
            "entire Wikipedia corpus and the BookCorpus dataset."
        ),
    },
    {
        "question": "What is the key innovation in the Transformer architecture?",
        "ground_truth": (
            "The key innovation is the multi-head attention mechanism, which allows "
            "the model to jointly attend to information from different representation "
            "subspaces, replacing traditional recurrent neural networks."
        ),
    },
    {
        "question": "How does RAG combine parametric and non-parametric memory?",
        "ground_truth": (
            "RAG combines parametric memory from a pre-trained language model with "
            "non-parametric memory via a dense retrieval component (DPR) that fetches "
            "relevant documents from an external knowledge base."
        ),
    },
    {
        "question": "What metrics does RAGAS use to evaluate RAG pipelines?",
        "ground_truth": (
            "RAGAS measures faithfulness, answer relevancy, context recall, and "
            "context precision to evaluate RAG pipeline quality."
        ),
    },
    {
        "question": "What is Ollama and which models does it support?",
        "ground_truth": (
            "Ollama is an open-source tool for running large language models locally. "
            "It supports models including Llama 3, Mistral, Gemma, Phi, Qwen, and DeepSeek."
        ),
    },
    {
        "question": "What is the difference between GPT-2 and GPT-3?",
        "ground_truth": (
            "GPT-2 has 1.5 billion parameters and showed emergent text generation abilities. "
            "GPT-3 scaled to 175 billion parameters and demonstrated few-shot learning "
            "capabilities without any fine-tuning."
        ),
    },
    {
        "question": "What Python library is commonly used for knowledge graphs in research?",
        "ground_truth": (
            "NetworkX is a Python library widely used for constructing and analyzing "
            "graphs in research settings."
        ),
    },
    {
        "question": "What is LangChain and who created it?",
        "ground_truth": (
            "LangChain is an open-source framework for building applications powered "
            "by large language models, created by Harrison Chase and first released in 2022."
        ),
    },
]


# ---------------------------------------------------------------------------
# RAGASEvaluator
# ---------------------------------------------------------------------------
class RAGASEvaluator:
    """
    Runs RAGAS evaluation over the Graph RAG pipeline.

    Usage:
        evaluator = RAGASEvaluator(pipeline)
        results = evaluator.evaluate()
        evaluator.print_report(results)
        evaluator.save_report(results)
    """

    def __init__(
        self,
        pipeline,
        eval_dataset: Optional[list] = None,
        judge_model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self.pipeline = pipeline
        self.eval_dataset = eval_dataset or DEFAULT_EVAL_DATASET
        self.judge_model = judge_model
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Build evaluation dataset
    # ------------------------------------------------------------------
    def _build_ragas_dataset(self) -> Dataset:
        """
        Runs the pipeline over each eval question and assembles the
        HuggingFace Dataset that RAGAS expects.
        """
        questions, answers, contexts, ground_truths = [], [], [], []

        logger.info("Running pipeline over %d evaluation questions…", len(self.eval_dataset))

        for item in self.eval_dataset:
            question     = item["question"]
            ground_truth = item["ground_truth"]

            result = self.pipeline.ask(question)

            questions.append(question)
            answers.append(result["answer"])
            contexts.append([doc.page_content for doc in result["source_docs"]])
            ground_truths.append(ground_truth)

            logger.info("  Q: %s", question[:70])
            logger.info("  A: %s", result["answer"][:100])

        return Dataset.from_dict(
            {
                "question":     questions,
                "answer":       answers,
                "contexts":     contexts,
                "ground_truth": ground_truths,
            }
        )

    # ------------------------------------------------------------------
    # Manual scoring fallback (when RAGAS import fails)
    # ------------------------------------------------------------------
    def _manual_score(self, dataset: Dataset) -> dict:
        """
        Simple heuristic scoring when RAGAS is unavailable.
        Uses keyword overlap as a proxy for each metric.
        """
        logger.warning("RAGAS not available — using heuristic scoring fallback.")

        def overlap(a: str, b: str) -> float:
            a_words = set(a.lower().split())
            b_words = set(b.lower().split())
            if not b_words:
                return 0.0
            return len(a_words & b_words) / len(b_words)

        faithfulness_scores, relevancy_scores, recall_scores, precision_scores = [], [], [], []

        for i in range(len(dataset)):
            question    = dataset[i]["question"]
            answer      = dataset[i]["answer"]
            contexts    = dataset[i]["contexts"]
            ground_truth = dataset[i]["ground_truth"]

            ctx_combined = " ".join(contexts)

            faithfulness_scores.append(overlap(answer, ctx_combined))
            relevancy_scores.append(overlap(answer, question))
            recall_scores.append(overlap(ctx_combined, ground_truth))
            precision_scores.append(overlap(ctx_combined, answer))

        def avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        scores = {
            "faithfulness":      avg(faithfulness_scores),
            "answer_relevancy":  avg(relevancy_scores),
            "context_recall":    avg(recall_scores),
            "context_precision": avg(precision_scores),
        }
        scores["composite"] = round(sum(scores.values()) / 4, 4)
        return scores

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        """
        Runs evaluation and returns a results dict with scores.
        Uses RAGAS if available, falls back to heuristic scoring otherwise.
        """
        dataset = self._build_ragas_dataset()

        if RAGAS_AVAILABLE:
            try:
                logger.info("Running RAGAS evaluation…")

                # Configure RAGAS to use local Ollama LLM
                from langchain_ollama import OllamaLLM, OllamaEmbeddings

                llm = OllamaLLM(model=self.judge_model, base_url=self.base_url, temperature=0)
                embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=self.base_url)

                # Inject into metrics
                metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
                for metric in metrics:
                    if hasattr(metric, "llm"):
                        metric.llm = llm
                    if hasattr(metric, "embeddings"):
                        metric.embeddings = embeddings

                ragas_result = ragas_evaluate(dataset=dataset, metrics=metrics)

                scores = {
                    "faithfulness":      round(float(ragas_result["faithfulness"]),      4),
                    "answer_relevancy":  round(float(ragas_result["answer_relevancy"]),  4),
                    "context_recall":    round(float(ragas_result["context_recall"]),    4),
                    "context_precision": round(float(ragas_result["context_precision"]), 4),
                }
                scores["composite"] = round(sum(scores.values()) / 4, 4)

            except Exception as exc:
                logger.warning("RAGAS evaluation failed (%s) — using heuristic fallback.", exc)
                scores = self._manual_score(dataset)
        else:
            scores = self._manual_score(dataset)

        return {
            "scores":  scores,
            "dataset": dataset,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_report(self, results: dict) -> None:
        scores = results["scores"]

        print("\n" + "=" * 58)
        print("  GRAPH RAG PIPELINE — EVALUATION REPORT")
        print("=" * 58)
        print(f"  {'Metric':<25} {'Score':>8}  {'Bar'}")
        print("-" * 58)

        for metric, score in scores.items():
            bar_len = int(score * 20)
            bar   = "█" * bar_len + "░" * (20 - bar_len)
            flag  = " ✓" if score >= 0.7 else (" ⚠" if score >= 0.5 else " ✗")
            print(f"  {metric:<25} {score:>6.3f}  [{bar}]{flag}")

        print("=" * 58)
        print(f"  Evaluated on {len(self.eval_dataset)} questions")
        print("=" * 58 + "\n")

    def save_report(self, results: dict, output_path: str = "data/evaluation_report.json") -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp":       datetime.now().isoformat(),
            "scores":          results["scores"],
            "num_questions":   len(self.eval_dataset),
            "eval_questions":  [item["question"] for item in self.eval_dataset],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", output_path)

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------
    @staticmethod
    def suggest_improvements(scores: dict) -> list:
        suggestions = []
        if scores.get("faithfulness", 1.0) < 0.7:
            suggestions.append(
                "Low faithfulness: Try lower temperature or stronger context-grounding in the prompt."
            )
        if scores.get("answer_relevancy", 1.0) < 0.7:
            suggestions.append(
                "Low answer relevancy: Try query rewriting or increase chunk overlap."
            )
        if scores.get("context_recall", 1.0) < 0.7:
            suggestions.append(
                "Low context recall: Increase retrieval_k or reduce chunk size."
            )
        if scores.get("context_precision", 1.0) < 0.7:
            suggestions.append(
                "Low context precision: Reduce retrieval_k or add a re-ranker."
            )
        if not suggestions:
            suggestions.append(
                "All metrics above 0.7 — good pipeline performance!"
            )
        return suggestions
