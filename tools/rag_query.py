import json
import os
from typing import Any, Dict, List, Optional

# Updated imports for Google Generative AI
from google import genai
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from embedchain import App

# Import new tools
from tools.hyde_rewriter import HydeRewriterTool
from tools.reranker import RerankerTool


class Source(BaseModel):
    document_id: str = Field(...)
    path: str = Field(...)
    page: Optional[int] = Field(None)
    chunk: Optional[int] = Field(None)
    content_type: str = Field(...)
    document_type: str = Field(...)
    relevance_score: float = Field(...)
    snippet: str = Field(...)
    image_path: Optional[str] = Field(None)
    image_width: Optional[int] = Field(None)
    image_height: Optional[int] = Field(None)


class QueryResult(BaseModel):
    query: str = Field(...)
    answer: str = Field(...)
    sources: List[Source] = Field(...)


class QueryRequest(BaseModel):
    queries: List[str] = Field(...)
    max_sources: int = Field(3)
    filters: Optional[Dict[str, Any]] = Field(None)


class QueryResponse(BaseModel):
    results: List[QueryResult] = Field(...)


class AnswerQueryTool(BaseTool):
    """Tool to search and answer queries using the document knowledge base.

    This tool integrates with the following components:
    - Embedchain for vector retrieval
    - Hypothetical Document Embeddings (HyDE) for query expansion
    - Cross-encoder reranking for improving relevance
    """

    name: str = "Answer Query Tool"
    description: str = "Tool to search and answer queries using the knowledge base"

    def __init__(self):
        """Initialize the AnswerQueryTool with optional HyDE and reranker components."""
        super().__init__(
            name="Answer Query Tool",
            description="Tool to search and answer queries using the knowledge base",
        )

        # Store helper tools in a dictionary for clean access
        self._tools = {}  # Dictionary to store tool instances

        # Initialize the HyDE rewriter tool
        try:
            self._tools["hyde_rewriter"] = HydeRewriterTool()
            print("HyDE rewriter initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize HyDE rewriter: {e}")

        # Initialize the reranker tool
        try:
            self._tools["reranker"] = RerankerTool()
            print("Reranker initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize reranker: {e}")

    def get_hyde_rewriter(self):
        """Get the HyDE rewriter tool instance if available."""
        return self._tools.get("hyde_rewriter")

    def get_reranker(self):
        """Get the reranker tool instance if available."""
        return self._tools.get("reranker")

    def _init_gemini_llm(self):
        """Initializes the Google Generative AI client."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Create and return a proper client instance using the new API
        return genai.Client(api_key=api_key)

    def _get_embedchain_config(self) -> Dict[str, Any]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        return {
            "llm": {
                "provider": "google",
                "config": {
                    "model": "gemini-2.0-flash",
                    "api_key": api_key,
                },
            },
            "embedder": {
                "provider": "google",
                "config": {
                    "model": "models/embedding-001",
                },
            },
            "vectordb": {
                "provider": "chroma",
                "config": {"collection_name": "document_qa", "dir": "db"},
            },
        }

    def _get_embedchain_app(self) -> App:
        config = self._get_embedchain_config()
        return App.from_config(config=config)

    def _parse_source(self, source_tuple) -> Source:
        document_content, metadata = source_tuple

        snippet = (
            document_content[:500] + "..."
            if len(document_content) > 500
            else document_content
        )

        # Get the path from either pdf_path or path
        path = metadata.get("pdf_path", metadata.get("path", "unknown"))

        # Get the page from either page_number or page
        page = metadata.get("page_number", metadata.get("page"))

        # Get image path from either page_image, embedded_image, or image_path
        image_path = metadata.get(
            "page_image", metadata.get("embedded_image", metadata.get("image_path"))
        )

        source = Source(
            document_id=metadata.get("document_id", "unknown"),
            path=path,
            page=page,
            chunk=metadata.get("chunk"),
            content_type=metadata.get("content_type", "text"),
            document_type=metadata.get("document_type", "unknown"),
            relevance_score=metadata.get("score", 0.0),
            snippet=snippet,
            image_path=image_path,
            image_width=metadata.get("image_width"),
            image_height=metadata.get("image_height"),
        )

        return source

    def _process_query(
        self, query: str, max_sources: int, filters: Dict
    ) -> QueryResult:
        ec_app = self._get_embedchain_app()

        try:
            # Apply HyDE query expansion if available
            original_query = query
            hyde_rewriter = self.get_hyde_rewriter()
            if hyde_rewriter is not None:
                try:
                    hyde_request = {
                        "query": query,
                        "context_length": 500,  # Use longer context
                        "output_format": "text",
                    }
                    expanded_query = hyde_rewriter._run(json.dumps(hyde_request))
                    if expanded_query and expanded_query.strip():
                        print("Query expanded with HyDE rewriter")
                        # Use the expanded query for retrieval but keep original for final response
                        query = expanded_query
                except Exception as e:
                    print(f"Error in HyDE query expansion: {e}")
                    # Continue with original query if HyDE fails
                    pass

            # Execute query with filters
            if filters:
                result = ec_app.query(query, citations=True, where=filters)
            else:
                result = ec_app.query(query, citations=True)

            embedchain_answer = ""
            citations = []

            if isinstance(result, tuple) and len(result) == 2:
                embedchain_answer = result[0]
                citations = result[1]
            elif isinstance(result, dict) and "response" in result:
                embedchain_answer = result["response"]
                citations = result.get("citations", [])
            else:
                embedchain_answer = str(result)
                citations = []

            # Apply reranking if available
            reranker = self.get_reranker()
            if reranker is not None and citations:
                try:
                    # Prepare sources for reranking
                    sources_for_reranking = []
                    for source in citations:
                        content, metadata = source
                        sources_for_reranking.append(
                            {
                                "content": content,
                                "metadata": metadata,
                                "score": metadata.get("score", 0.0),
                            }
                        )

                    # Prepare reranker request
                    rerank_request = {
                        "original_query": original_query,  # Use original query for relevance judgment
                        "results": sources_for_reranking,
                        "top_n": max_sources,
                    }

                    # Execute reranking
                    rerank_result_json = reranker._run(json.dumps(rerank_request))
                    rerank_result = json.loads(rerank_result_json)

                    if (
                        "reranked_results" in rerank_result
                        and rerank_result["reranked_results"]
                    ):
                        print("Sources reranked successfully")
                        # Transform reranked results back to expected format
                        reranked_citations = []
                        for item in rerank_result["reranked_results"]:
                            content = item["content"]
                            metadata = item["metadata"]
                            # Add rerank score to metadata
                            metadata["score"] = item["rerank_score"]
                            reranked_citations.append((content, metadata))

                        # Replace citations with reranked results
                        citations = reranked_citations
                except Exception as e:
                    print(f"Error during reranking: {e}")
                    # Continue with original citations if reranking fails
                    pass

            parsed_sources = []
            source_context = ""

            # Take only up to max_sources
            for source in citations[:max_sources]:
                content, metadata = source
                parsed_source = self._parse_source(source)
                parsed_sources.append(parsed_source)

                doc_type = metadata.get("document_type", "unknown")
                # Get path from either pdf_path or path
                path = metadata.get("pdf_path", metadata.get("path", "unknown"))
                # Get page from either page_number or page
                page = metadata.get("page_number", metadata.get("page"))
                page_info = f" (page {page})" if page else ""

                source_context += (
                    f"\n--- Source from {doc_type} document: {path}{page_info} ---\n"
                )
                source_context += content + "\n"

            answer = embedchain_answer
            if source_context:
                # Use the Gemini client to generate a comprehensive answer based on the sources
                client = self._init_gemini_llm()
                try:
                    # Create the prompt for answer generation
                    user_prompt = f"""Question: {query}

Sources:
{source_context}

Please provide an extremely comprehensive, detailed, and thorough answer to the question based solely on the provided document sources. Your answer should:

1. Be extensive and informative, capturing all relevant information from the sources
2. Include all important details, examples, and context from the documents
3. Use direct quotes and specific information from the sources to support your answer
4. Be well-structured with logical organization and clear transitions
5. Include specific facts, figures, and data points mentioned in the sources when relevant
6. Synthesize information from multiple sources when applicable

The goal is to create the most complete and detailed answer possible while remaining accurate to the source material. If the sources don't contain enough information to fully answer the question, acknowledge the specific limitations and explain what additional information would be needed."""

                    # Generate content with the new GenAI client
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[
                            {"role": "user", "parts": [{"text": user_prompt}]},
                        ],
                    )

                    # Use the generated response if available
                    if (
                        response
                        and hasattr(response, "text")
                        and response.text is not None
                    ):
                        answer = response.text.strip()
                except Exception as e:
                    print(f"Error enhancing answer with Gemini: {e}")
                    # Fall back to embedchain answer if enhancement fails

            return QueryResult(
                query=query, answer=answer, sources=parsed_sources
            )

        except Exception as e:
            return QueryResult(
                query=query, answer=f"Error processing query: {str(e)}", sources=[]
            )

    def _run(self, request_json: str) -> str:
        try:
            # Input validation to ensure request_json is actually a string
            if not isinstance(request_json, str):
                error_msg = f"Invalid input type: expected string, got {type(request_json).__name__}"
                print(f"Input error: {error_msg}")
                error_response = QueryResponse(
                    results=[
                        QueryResult(
                            query="Error",
                            answer=f"Error processing request: {error_msg}. Please provide request_json as a properly formatted JSON string.",
                            sources=[],
                        )
                    ]
                )
                return json.dumps(error_response.model_dump())

            # Try to parse the JSON string
            try:
                request_data = json.loads(request_json)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON format: {str(e)}"
                print(f"JSON parse error: {error_msg}")
                error_response = QueryResponse(
                    results=[
                        QueryResult(
                            query="Error",
                            answer=f"Error processing request: {error_msg}. Please provide a valid JSON string.",
                            sources=[],
                        )
                    ]
                )
                return json.dumps(error_response.model_dump())

            # Validate the request object
            request = QueryRequest(**request_data)

            results = []

            for query in request.queries:
                result = self._process_query(
                    query=query,
                    max_sources=request.max_sources,
                    filters=request.filters or {},
                )
                results.append(result)

            response = QueryResponse(results=results)

            return json.dumps(response.model_dump())

        except Exception as e:
            error_msg = str(e)
            print(f"Processing error: {error_msg}")
            error_response = QueryResponse(
                results=[
                    QueryResult(
                        query="Error",
                        answer=f"Error processing request: {error_msg}",
                        sources=[],
                    )
                ]
            )
            return json.dumps(error_response.model_dump())
