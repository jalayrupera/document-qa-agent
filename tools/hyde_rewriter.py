"""
Hypothetical Document Embeddings (HyDE) Rewriter for RAG

This module provides HyDE functionality to improve RAG query performance
by rewriting user queries into hypothetical document content.
"""

import json
import os

# Updated imports for Google Generative AI
from google import genai
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class HydeRewriteRequest(BaseModel):
    """Input schema for HyDE rewrite"""

    query: str = Field(..., description="Original user query to rewrite")
    context_length: int = Field(
        300,
        description="Target length for generated hypothetical document",
        ge=100,
        le=1000,
    )
    output_format: str = Field(
        "text",
        description="Output format: 'text' or 'json' (includes both original and rewritten)",
    )


class HydeRewriteResponse(BaseModel):
    """Output schema for HyDE rewrite"""

    original_query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(..., description="Rewritten/expanded query")


class HydeRewriterTool(BaseTool):
    """
    Implements Hypothetical Document Embeddings (HyDE) approach by
    rewriting user queries into hypothetical document content that would
    contain the answer to the query.
    """

    name: str = "HyDE Rewriter Tool"
    description: str = "Tool to rewrite user queries into hypothetical document content"

    def __init__(self):
        super().__init__(
            name="HyDE Rewriter Tool",
            description="Tool to rewrite user queries into hypothetical document content",
        )

    def _init_gemini_client(self):
        """Initialize the Gemini model client for text generation"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        return genai.Client(api_key=api_key)

    def rewrite_query(self, query: str, context_length: int = 150) -> str:
        """
        Rewrite a user query into hypothetical document content

        Args:
            query: Original user query
            context_length: Target length for the generated content

        Returns:
            Rewritten query as hypothetical document content
        """
        client = self._init_gemini_client()

        user_prompt = f"""Generate a short, factual document passage (around 150 words) that would answer this question:
        
Question: {query}

Important instructions:
1. Be factual and precise - avoid generating speculative content
2. Keep the answer concise and focused on the query
3. Use simple, direct language
4. Do not embellish or add information beyond what would be in a standard reference
5. Focus only on core facts that would appear in a technical document

The passage should read like it comes from an authoritative reference manual or technical document."""

        try:
            # Generate content according to the Google GenAI API documentation
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
            )

            # Extract text from response
            if response and hasattr(response, "text") and response.text is not None:
                rewritten_query = response.text

                # Clean up the response (remove any markdown formatting, etc.)
                rewritten_query = rewritten_query.strip()
                if rewritten_query and rewritten_query.startswith("Document passage:"):
                    rewritten_query = rewritten_query[
                        len("Document passage:") :
                    ].strip()

                return rewritten_query
            else:
                print("Error: Empty or invalid response from Gemini")
                return query

        except Exception as e:
            print(f"Error generating hypothetical document: {e}")
            # Return original query as fallback
            return query

    def _run(self, request_json: str) -> str:
        """
        Execute the HyDE rewriting action

        Args:
            request_json: JSON string with the rewrite request

        Returns:
            JSON string with the rewriting results
        """
        try:
            # Parse request
            request_data = json.loads(request_json)

            # Extract parameters
            query = request_data.get("query", "")
            context_length = int(request_data.get("context_length", 300))
            output_format = request_data.get("output_format", "text")

            # Generate the hypothetical document content
            rewritten_query = self.rewrite_query(query, context_length)

            # Format response based on output_format
            if output_format == "json":
                response = {"original_query": query, "rewritten_query": rewritten_query}
                return json.dumps(response)
            else:
                # Just return the rewritten text for direct use
                return rewritten_query

        except Exception as e:
            print(f"Error in HyDE rewriting: {e}")
            # Return empty or error json
            return json.dumps(
                {
                    "original_query": request_data.get("query", ""),
                    "rewritten_query": "",
                    "error": str(e),
                }
            )
