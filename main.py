import ast
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
from crewai import LLM, Agent, Crew, Process, Task

from tools.rag_ingest import IngestDataTool
from tools.rag_query import AnswerQueryTool

dotenv.load_dotenv()

GEMINI_MODEL = "gemini/gemini-2.0-flash"
TEMPERATURE = 0.5


def get_document_type(path: str) -> str:
    if os.path.isdir(path):
        return "folder"

    file_extension = Path(path).suffix.lower()
    if file_extension == ".pdf":
        return "pdf"
    elif file_extension == ".docx":
        return "docx"
    elif file_extension == ".md":
        return "md"
    else:
        return "txt"


def extract_json_from_response(result_str: str) -> Optional[Dict]:
    if "```" in result_str:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_str)
        if match:
            result_str = match.group(1).strip()

    parsing_methods = [
        lambda s: json.loads(s),
        lambda s: json.loads(s.strip()),
        lambda s: ast.literal_eval(s),
    ]

    for parse_method in parsing_methods:
        try:
            return parse_method(result_str)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

    return None


def display_formatted_answer(answer: str, sources: Optional[List[Dict]] = None) -> None:
    terminal_width = os.get_terminal_size().columns
    width = min(terminal_width, 100)

    print("\n" + "=" * width)
    print("ANSWER")
    print("-" * width)

    wrapper = textwrap.TextWrapper(width=width, initial_indent="", subsequent_indent="")
    for line in wrapper.wrap(answer):
        print(line)

    if not sources:
        print("\n" + "=" * width)
        return

    print("\n" + "-" * width)
    print("SOURCES")
    print("-" * width)

    for i, source in enumerate(sources):
        # Get the path from either pdf_path or path
        path = source.get("pdf_path", source.get("path", "Unknown"))

        # Get the page from either page_number or page
        page = source.get("page_number", source.get("page"))

        # Get image path from either page_image, embedded_image, or image_path
        image_path = source.get(
            "page_image", source.get("embedded_image", source.get("image_path"))
        )

        content_type = source.get("content_type", "text")
        snippet = source.get("snippet", "")

        print(f"\n[{i + 1}]")

        source_wrapper = textwrap.TextWrapper(
            width=width, initial_indent="    ", subsequent_indent="    "
        )

        for line in source_wrapper.wrap(snippet):
            print(line)

        page_info = f" (Page: {page})" if page is not None else ""
        print(f"    Source: {path}{page_info}")
        print(f"    Type: {content_type}")

        if content_type == "image" and image_path:
            print(f"    Image: {image_path}")

        print("    " + "-" * (width - 4))

    print("\n" + "=" * width)


def check_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "Please set GEMINI_API_KEY environment variable. "
            "Get a key from https://makersuite.google.com/app/apikey"
        )


def create_tools():
    ingest_tool = IngestDataTool()
    query_tool = AnswerQueryTool()
    # The HyDE rewriter and reranker tools are now initialized in the AnswerQueryTool
    # This ensures they are properly integrated into the RAG pipeline
    return ingest_tool, query_tool


def create_agents(ingest_tool, query_tool, llm):
    ingestion_agent = Agent(
        role="Document Ingestion Specialist",
        goal="Efficiently process documents and add their content to the knowledge base",
        backstory="Expert in document ingestion and processing for knowledge retrieval. "
        "You extract text and images from various document formats and "
        "store them with proper metadata for later retrieval.",
        verbose=True,
        allow_delegation=False,
        tools=[ingest_tool],
        llm=llm,
    )

    query_agent = Agent(
        role="Advanced RAG Query Agent",
        goal="Answer questions accurately using the knowledge base",
        backstory="Expert in information retrieval and question answering. "
        "You search through a knowledge base to find relevant information "
        "and generate comprehensive answers with source citations. "
        "You are skilled at properly formatting data for tools and always ensure that JSON "
        "strings are correctly formatted when passing to tools. You always serialize request_json as "
        "a valid JSON string before passing it to tools.",
        verbose=True,
        allow_delegation=False,
        tools=[query_tool],
        llm=llm,
    )

    return ingestion_agent, query_agent


def process_document(document_path: str, ingestion_agent: Agent) -> str:
    doc_type = get_document_type(document_path)

    ingest_request = {
        "paths": [document_path],
        "metadata": {
            "source_type": doc_type,
            "processed_at": str(Path(document_path).stat().st_mtime),
        },
    }

    ingest_json = json.dumps(ingest_request)

    process_doc_task = Task(
        description=f"Process the {doc_type} at: {document_path}. Use the following JSON input: {ingest_json}",
        expected_output="JSON response with ingestion results including processed files, document IDs, and total chunks created.",
        agent=ingestion_agent,
    )

    process_crew = Crew(
        agents=[ingestion_agent],
        tasks=[process_doc_task],
        process=Process.sequential,
        verbose=True,
    )

    result = process_crew.kickoff()
    return str(result[0]) if isinstance(result, list) else str(result)


def process_query(query: str, query_agent: Agent) -> Dict:
    query_request = {
        "queries": [query],
        "max_sources": 5,
        "filters": {},
    }

    query_json = json.dumps(query_request)

    query_task = Task(
        description=f"Answer this question: '{query}'. Use the following JSON input as a STRING when calling the Answer Query Tool: {query_json}. IMPORTANT: Make sure to properly JSON-encode the request_json parameter as a string when using the tool.",
        expected_output="JSON response with query results including answers and sources.",
        agent=query_agent,
    )

    query_crew = Crew(
        agents=[query_agent],
        tasks=[query_task],
        process=Process.sequential,
        verbose=True,
    )

    result = query_crew.kickoff()
    result_str = str(result[0]) if isinstance(result, list) else str(result)

    parsed_result = extract_json_from_response(result_str)

    if parsed_result and isinstance(parsed_result, dict):
        if "results" in parsed_result and len(parsed_result["results"]) > 0:
            query_result = parsed_result["results"][0]
            return {
                "response": query_result.get("answer", ""),
                "sources": query_result.get("sources", []),
            }
        return parsed_result
    else:
        return {"response": result_str, "sources": []}


def interactive_mode():
    print("===== Starting Document QA Agent =====")
    print("Initializing Agents and Tools...")

    try:
        check_api_key()

        llm = LLM(
            model=GEMINI_MODEL,
            temperature=0.5,
            api_key=os.getenv("GEMINI_API_KEY", ""),
        )

        ingest_tool, query_tool = create_tools()

        ingestion_agent, query_agent = create_agents(ingest_tool, query_tool, llm)
        print("Initialization complete.")

        # Print information about the advanced RAG features
        print("\nAdvanced RAG features enabled:")
        print(
            "- HyDE (Hypothetical Document Embeddings): Improves retrieval by expanding queries"
        )
        print("- Cross-encoder Reranking: Improves result relevance ordering")

    except Exception as e:
        print(f"FATAL: Failed to initialize components: {e}")
        return

    print("\nProcess documents/folders and ask questions about the knowledge base.")
    print(
        "Type 'process' to add a document/folder, 'query' to ask a question, 'exit' to quit."
    )

    mode = "process"

    while True:
        if mode == "process":
            document_path = input(
                "\nEnter document or folder path to process (or type 'query', 'exit'): "
            ).strip()

            if document_path.lower() == "exit":
                break
            elif document_path.lower() == "query":
                mode = "query"
                continue
            elif not document_path:
                continue

            if not os.path.exists(document_path):
                print(f"Error: Path '{document_path}' does not exist.")
                continue

            print(f"\nProcessing '{document_path}'...")
            try:
                result = process_document(document_path, ingestion_agent)
                print("\n===== Document Processing Result =====")
                print(result)
            except Exception as e:
                print(f"\nError processing document: {e}")
                continue

        elif mode == "query":
            query = input("\nEnter your query (or type 'process', 'exit'): ").strip()

            if query.lower() == "exit":
                break
            elif query.lower() == "process":
                mode = "process"
                continue
            elif not query:
                continue

            print(f"\nAnswering: {query}\n")
            print("Processing query... This may take a moment.")

            try:
                result = process_query(query, query_agent)

                answer = result.get("response", "No answer provided")
                sources = result.get("sources", [])
                display_formatted_answer(answer, sources)

            except Exception as e:
                print(f"\nError during query: {e}")


def main():
    interactive_mode()


if __name__ == "__main__":
    main()
