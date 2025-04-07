import json
import os
from typing import Any, Dict, List, Optional

import google.genai as genai
from google.genai import Client
from crewai.tools import BaseTool
from embedchain import App
from pydantic import BaseModel, Field


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
    name: str = "Answer Query Tool"
    description: str = "Tool to search and answer queries using the knowledge base"

    def __init__(self):
        super().__init__(name="Answer Query Tool", description="Tool to search and answer queries using the knowledge base")

    def _init_gemini_llm(self) -> Client:
        """Initializes the Google Generative AI model."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

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

        source = Source(
            document_id=metadata.get("document_id", "unknown"),
            path=metadata.get("path", "unknown"),
            page=metadata.get("page"),
            chunk=metadata.get("chunk"),
            content_type=metadata.get("content_type", "text"),
            document_type=metadata.get("document_type", "unknown"),
            relevance_score=metadata.get("score", 0.0),
            snippet=snippet,
            image_path=metadata.get("image_path"),
            image_width=metadata.get("image_width"),
            image_height=metadata.get("image_height"),
        )

        return source

    def _process_query(
        self, query: str, max_sources: int, filters: Dict
    ) -> QueryResult:
        ec_app = self._get_embedchain_app()

        gemini_llm = self._init_gemini_llm()

        try:
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

            parsed_sources = []
            source_context = ""

            for source in citations[:max_sources]:
                content, metadata = source
                parsed_source = self._parse_source(source)
                parsed_sources.append(parsed_source)

                doc_type = metadata.get("document_type", "unknown")
                path = metadata.get("path", "unknown")
                page_info = (
                    f" (page {metadata.get('page')})" if metadata.get("page") else ""
                )

                source_context += (
                    f"\n--- Source from {doc_type} document: {path}{page_info} ---\n"
                )
                source_context += content + "\n"

            answer = embedchain_answer
            if source_context:
                prompt = f"""Based on the following information, please answer the question: "{query}"
                
{source_context}

Generate a comprehensive, accurate answer based only on the information provided above.
If the information doesn't contain a complete answer, acknowledge the limitations of what can be determined from the available sources.
"""
                # Generate content using the native client
                response = gemini_llm.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )

                # Extract text from the response
                llm_answer = response.text if response.text else ""
                llm_answer = llm_answer.strip()

                if llm_answer:
                    answer = llm_answer
            else:
                answer = "No relevant information found to answer the query."

            query_result = QueryResult(
                query=query, answer=answer, sources=parsed_sources
            )

            return query_result

        except Exception as e:
            return QueryResult(
                query=query, answer=f"Error processing query: {str(e)}", sources=[]
            )

    def _run(self, request_json: str) -> str:
        try:
            request_data = json.loads(request_json)
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
            error_response = QueryResponse(
                results=[
                    QueryResult(
                        query="Error",
                        answer=f"Error processing request: {str(e)}",
                        sources=[],
                    )
                ]
            )
            return json.dumps(error_response.model_dump())
