import sys
import json
import re

from typing import List, Set, Dict, Any, Optional, Sequence
from datetime import datetime

from langchain_openai import ChatOpenAI

from src.exception import CustomException
from src.logger import setup_logger


logger = setup_logger()


class RAGEvaluator:
    """
    Evaluate retrieval and generation quality for the Aurora RAG system.
    """
    def __init__(self, llm: Optional[ChatOpenAI] = None, model_name: str = "gpt-5.4-mini"):
        try:
            if llm is None:
                logger.info( "Creating evaluation LLM with model: %s", model_name)

                self.llm = ChatOpenAI(model=model_name, temperature=0)

            else:
                self.llm = llm

            logger.info("RAGEvaluator initialized successfully.")

        except Exception as e:
            logger.error("Failed to initialize RAGEvaluator.")
            raise CustomException(e, sys) from e

    # RETRIEVAL METRICS

    def compute_recall_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
        """
        Calculate the proportion of all relevant documents that
        appear within the top-k retrieved documents.
        """

        try:
            if k <= 0:
                return 0.0

            if not relevant_docs:
                return 0.0

            retrieved_at_k = retrieved_docs[:k]

            relevant_retrieved = len(set(retrieved_at_k).intersection(relevant_docs))

            score = (relevant_retrieved / len(relevant_docs))

            logger.debug("Recall@%s: %.3f", k, score)

            return score

        except Exception as e:
            logger.error("Error computing Recall@%s.", k)
            return 0.0

    def compute_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
        """
        Calculate the reciprocal rank of the first relevant
        document within the top-k retrieved results.
        """

        try:
            if k <= 0 or not relevant_docs:
                return 0.0

            retrieved_at_k = retrieved_docs[:k]

            for rank, document in enumerate(retrieved_at_k, start=1):
                if document in relevant_docs:
                    score = 1.0 / rank
                    logger.debug("Reciprocal Rank@%s: %.3f", k, score)

                    return score
            return 0.0

        except Exception:
            logger.error("Error computing Reciprocal Rank@%s.", k)
            return 0.0

    def compute_precision_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
        """
        Calculate the proportion of the top-k retrieved documents
        that are relevant.
        """

        try:
            if k <= 0:
                return 0.0

            retrieved_at_k = retrieved_docs[:k]

            if not retrieved_at_k:
                return 0.0

            relevant_retrieved = sum(document in relevant_docs for document in retrieved_at_k)

            score = relevant_retrieved / k

            logger.debug("Precision@%s: %.3f", k, score)

            return score

        except Exception:
            logger.error("Error computing Precision@%s.", k)
            return 0.0

   
    # GENERATION METRICS

    def _extract_score(self, model_output: str) -> float:
        """
        Extract a numeric evaluation score between 0 and 1
        from an LLM response.
        """

        match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", model_output.strip())

        if not match:
            raise ValueError(f"No valid score found in model output: {model_output}")

        score = float(match.group())

        return min(max(score, 0.0), 1.0)

    def evaluate_faithfulness(self, response: str, context: str, verbose: bool = False) -> float:
        """
        Evaluate whether the generated response is supported
        by the retrieved context.
        """

        try:
            prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Rate how well the response is supported by the provided context.

Context:
{context[:4000]}

Response:
{response}

Return only one numeric score between 0 and 1.

1 = Completely supported by the context
0 = Not supported by the context

Score:
""".strip()

            result = self.llm.invoke(prompt)

            score = self._extract_score(result.content)

            if verbose:
                logger.info("Faithfulness score: %.3f", score)

            return score

        except Exception as e:
            logger.error("Error evaluating faithfulness: %s", e)

            return 0.5

    def evaluate_relevance(self, response: str, question: str, verbose: bool = False) -> float:
        """
        Evaluate whether the generated response answers
        the user's question.
        """

        try:
            prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Rate how relevant the response is to the user's question.

Question:
{question}

Response:
{response}

Return only one numeric score between 0 and 1.

1 = Completely relevant
0 = Not relevant

Score:
""".strip()

            result = self.llm.invoke(prompt)

            score = self._extract_score(result.content)

            if verbose:
                logger.info("Relevance score: %.3f", score)

            return score

        except Exception as e:
            logger.error("Error evaluating relevance: %s", e)

            return 0.5

    def evaluate_response_quality(self, response: str, question: str, context: str, verbose: bool = False) -> Dict[str, float]:
        """
        Calculate faithfulness, relevance and overall
        response-quality scores.
        """
        try:
            faithfulness = (self.evaluate_faithfulness(response=response, context=context, verbose=verbose))

            relevance = (self.evaluate_relevance(response=response, question=question, verbose=verbose))

            overall = (faithfulness + relevance) / 2
            quality = {"faithfulness": faithfulness, "relevance": relevance, "overall": overall}

            logger.debug(
                "Response quality - Faithfulness: %.3f, Relevance: %.3f, Overall: %.3f", faithfulness, relevance, overall)
            return quality

        except Exception as e:
            logger.error("Error evaluating response quality: %s", e)
            return {"faithfulness": 0.5, "relevance": 0.5, "overall": 0.5}

  
    # COMBINED RETRIEVAL EVALUATION
   
    def evaluate_retrieval_quality(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        k_values: Sequence[int] = (1, 3, 5)
    ) -> Dict[str, float]:
        """
        Calculate retrieval metrics across multiple values of k.
        """

        try:
            metrics = {}

            for k in k_values:

                metrics[f"recall_at_{k}"] = self.compute_recall_at_k(
                    retrieved_docs=retrieved_docs,
                    relevant_docs=relevant_docs,
                    k=k
                )

                metrics[
                    f"precision_at_{k}"
                ] = self.compute_precision_at_k( retrieved_docs=retrieved_docs,  relevant_docs=relevant_docs,k=k)

            max_k = max(k_values)

            metrics["reciprocal_rank"] = (self.compute_reciprocal_rank(retrieved_docs=retrieved_docs, relevant_docs=relevant_docs, k=max_k))
            logger.debug("Retrieval quality metrics: %s", json.dumps(metrics, indent=2))
            return metrics

        except Exception as e:
            logger.error("Error evaluating retrieval quality: %s", e)
            return {}

    
    # EXPORT EVALUATION RESULTS

    def export_results(self, results: List[Dict[str, Any]], filepath: str = "evaluation_results.json") -> None:
        """
        Export RAG evaluation results to JSON.
        """

        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(results),
                "results": results}

            with open(filepath, "w", encoding="utf-8") as file:

                json.dump(export_data, file, indent=2, ensure_ascii=False, default=str)

            logger.info("Evaluation results exported to %s", filepath)

        except Exception as e:
            logger.error("Failed to export evaluation results.")
            raise CustomException(e,sys) from e