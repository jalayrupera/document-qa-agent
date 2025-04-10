"""
Reranker Module for Advanced RAG

This module provides functionality to re-rank retrieved document chunks
based on their relevance to the original query using a cross-encoder model.
"""

import ast
import json
import typing as t

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers.cross_encoder import CrossEncoder

# Constants
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 5

# Initialize the re-ranker model globally
try:
    reranker = CrossEncoder(DEFAULT_RERANKER_MODEL)
    print(f"Reranker model '{DEFAULT_RERANKER_MODEL}' initialized successfully")
except Exception as e:
    print(f"Warning: Failed to load reranker model: {e}")
    reranker = None


class RerankResultsInput(BaseModel):
    """Input schema for reranking results"""

    original_query: str = Field(
        ..., description="The original user query used for relevance scoring."
    )
    results: t.Any = Field(
        ...,
        description="The initial list of retrieved results (dicts with 'content', 'metadata', 'score').",
    )
    top_n: int = Field(
        DEFAULT_TOP_N,
        description="The number of results to return after re-ranking.",
        gt=0,
    )


class RerankResultOutputItem(BaseModel):
    """Schema for an individual reranked result item"""

    content: str
    metadata: t.Optional[t.Dict[str, t.Any]] = None
    rerank_score: float
    score: t.Optional[float] = None  # Original score from retrieval

    class Config:
        extra = "allow"


class RerankResultsOutput(BaseModel):
    """Output schema for reranking results"""

    reranked_results: t.List[RerankResultOutputItem] = Field(
        ...,
        description="The re-ranked and truncated list of results, including the rerank_score.",
    )


class RerankerTool(BaseTool):
    """
    Re-ranks a list of retrieved document chunks based on their relevance
    to the original query using a cross-encoder model.
    """

    name: str = "Reranker Tool"
    description: str = "Tool to re-rank RAG results based on query relevance"

    def __init__(self):
        super().__init__(
            name="Reranker Tool",
            description="Tool to re-rank RAG results based on query relevance",
        )

    def parse_results_input(self, raw_results: t.Any) -> t.List[t.Dict]:
        """
        Parse the results input which may be a string or list
        Args:
            raw_results: The raw results input (string, list, etc.)
        Returns:
            List of parsed result items as dictionaries
        """
        # Handle string input
        if isinstance(raw_results, str):
            try:
                # First try standard JSON parsing
                return json.loads(raw_results)
            except json.JSONDecodeError:
                try:
                    # If that fails, try ast.literal_eval for Python literal structures
                    return ast.literal_eval(raw_results)
                except (SyntaxError, ValueError) as e:
                    print(f"Error: Could not parse results string: {e}")
                    return []

        # Handle list input
        if isinstance(raw_results, list):
            return raw_results

        print(f"Error: Expected list for results, got {type(raw_results)}")
        return []

    def validate_result_item(self, item: t.Any) -> t.Optional[t.Dict]:
        """
        Validate and normalize a single result item
        Args:
            item: A result item to validate
        Returns:
            Validated and normalized result item, or None if invalid
        """
        try:
            if not isinstance(item, dict):
                print("Warning: Skipping non-dict result item")
                return None

            # Validate content
            content = item.get("content") or item.get("snippet")
            if not content or not isinstance(content, str):
                print("Warning: Skipping item with invalid content")
                return None

            # Handle metadata
            metadata = item.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    try:
                        metadata = ast.literal_eval(metadata)
                    except (SyntaxError, ValueError):
                        metadata = None

            if metadata and not isinstance(metadata, dict):
                metadata = None

            # Handle score
            score = item.get("score") or item.get("relevance_score")
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = None

            if score and not isinstance(score, (int, float)):
                score = None

            return {
                "content": content,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "score": score,
            }
        except Exception as e:
            print(f"Warning: Error processing result item: {e}")
            return None

    def rerank_results(
        self, original_query: str, valid_results: t.List[t.Dict], top_n: int
    ) -> t.List[t.Dict]:
        """
        Rerank the results using the cross-encoder model
        Args:
            original_query: The user's original query
            valid_results: List of validated result items
            top_n: Number of top results to return
        Returns:
            List of reranked results with scores
        """
        # Check if reranker is available
        if reranker is None:
            # Just sort by original score if available
            results_with_scores = [
                {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "rerank_score": result.get("score", 0.0),
                    "score": result.get("score"),
                }
                for result in valid_results
            ]
            results_with_scores.sort(key=lambda x: x["rerank_score"], reverse=True)
            return results_with_scores[:top_n]

        # Create pairs for the cross-encoder
        sentence_pairs = [
            (original_query, result["content"]) for result in valid_results
        ]

        # Predict scores
        scores = reranker.predict(sentence_pairs)

        # Combine scores with original results
        results_with_scores = []
        for i, result_dict in enumerate(valid_results):
            output_item = {
                "content": result_dict["content"],
                "metadata": result_dict["metadata"],
                "rerank_score": float(scores[i]),
                "score": result_dict.get("score"),
            }
            results_with_scores.append(output_item)

        # Sort by new score in descending order
        results_with_scores.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Truncate to top_n
        return results_with_scores[:top_n]

    def _run(self, request_json: str) -> str:
        """
        Execute the reranking action
        Args:
            request_json: The reranking request JSON containing query and results
        Returns:
            JSON string with reranked results
        """
        # Check if reranker is available
        if reranker is None:
            print("Error: Reranker model failed to load. Cannot execute reranking.")
            return json.dumps({"reranked_results": []})

        try:
            # Parse the request
            request_data = json.loads(request_json)

            # Extract request parameters
            original_query = request_data.get("original_query", "")
            raw_results = request_data.get("results", [])
            top_n = int(request_data.get("top_n", DEFAULT_TOP_N))

            # Parse and validate the results
            parsed_results = self.parse_results_input(raw_results)

            # Process and validate each result item
            valid_results = []
            for item in parsed_results:
                valid_item = self.validate_result_item(item)
                if valid_item:
                    valid_results.append(valid_item)

            if not valid_results:
                print("No valid results were processed from the input")
                return json.dumps({"reranked_results": []})

            # Rerank the results
            reranked_results = self.rerank_results(original_query, valid_results, top_n)

            return json.dumps({"reranked_results": reranked_results})

        except Exception as e:
            print(f"Error during re-ranking: {e}")
            return json.dumps({"reranked_results": [], "error": str(e)})
