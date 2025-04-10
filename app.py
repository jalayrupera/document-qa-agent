import json
import os
import tempfile
from pathlib import Path
from typing import Dict

import streamlit as st
from crewai import LLM, Agent, Task

from tools.rag_ingest import IngestDataTool
from tools.rag_query import AnswerQueryTool

st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        color: #1E88E5;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .section-header {
        color: #43A047;
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 8px;
    }
    
    .question-text {
        color: #6A1B9A;
        font-size: 18px;
        font-style: italic;
        padding: 10px;
        border-left: 3px solid #6A1B9A;
        background-color: #F3E5F5;
        margin-bottom: 15px;
    }
    
    .source-header {
        color: #FF6F00;
        font-size: 16px;
        font-weight: bold;
        margin-top: 12px;
    }
    
    .custom-divider {
        margin-top: 15px;
        margin-bottom: 15px;
        border-top: 1px solid #E0E0E0;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "GEMINI_API_KEY" not in os.environ and "api_key" not in st.session_state:
    st.sidebar.title("API Key Configuration")
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
    if st.sidebar.button("Save API Key"):
        if api_key:
            st.session_state["api_key"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key
            st.sidebar.success("API key saved!")
        else:
            st.sidebar.error("Please enter an API key")

    st.error("Please set your Gemini API Key in the sidebar to continue.")
    st.stop()
elif "api_key" in st.session_state and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = st.session_state["api_key"]

if "query_result" not in st.session_state:
    st.session_state.query_result = None


def get_document_type(path: str) -> str:
    """Determine document type based on file extension."""
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


@st.cache_resource
def initialize_tools_and_agents():
    """Initialize the LLM, tools and agents for document processing and querying."""
    try:
        llm = LLM(
            model="google/gemini-2.0-flash",
            temperature=0.5,
            api_key=os.environ.get("GEMINI_API_KEY", ""),
        )

        ingest_tool = IngestDataTool()
        query_tool = AnswerQueryTool()

        ingestion_agent = Agent(
            role="Document Ingestion Specialist",
            goal="Efficiently process documents and add their content to the knowledge base",
            backstory="Expert in document ingestion and processing for knowledge retrieval. You extract text and images from various document formats and store them with proper metadata for later retrieval.",
            verbose=True,
            allow_delegation=False,
            tools=[ingest_tool],
            llm=llm,
        )

        query_agent = Agent(
            role="Advanced RAG Query Agent",
            goal="Answer questions accurately using the knowledge base",
            backstory="Expert in information retrieval and question answering. You search through a knowledge base to find relevant information and generate comprehensive answers with source citations. You are skilled at properly formatting data for tools and always ensure that JSON strings are correctly formatted when passing to tools. You always serialize request_json as a valid JSON string before passing it to tools.",
            verbose=True,
            allow_delegation=False,
            tools=[query_tool],
            llm=llm,
        )

        return ingest_tool, query_tool, ingestion_agent, query_agent
    except Exception as e:
        st.error(f"Error initializing tools and agents: {e}")
        return None, None, None, None


def process_document(file_path: str, ingestion_agent: Agent) -> Dict:
    """Process a document using the ingestion agent."""
    doc_type = get_document_type(file_path)

    ingest_request = {
        "paths": [file_path],
        "metadata": {
            "source_type": doc_type,
            "processed_at": str(Path(file_path).stat().st_mtime),
        },
    }

    try:
        task = Task(
            description=f"Process the {doc_type} document at: {file_path}. Use the following JSON input: {json.dumps(ingest_request)}",
            expected_output="JSON response with ingestion results including processed files, document IDs, and total chunks created.",
            agent=ingestion_agent,
        )

        result = ingestion_agent.execute_task(task)

        try:
            if "```" in result:
                import re

                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result)
                if match:
                    result = match.group(1).strip()

            result_dict = json.loads(result)
            return result_dict
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"status": "completed", "message": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def process_query(query: str, query_agent: Agent) -> Dict:
    """Process a query using the query agent."""
    query_request = {
        "queries": [query],
        "max_sources": 5,
        "filters": {},
    }

    try:
        task = Task(
            description=f"Answer this question: '{query}'. Use the following JSON input as a STRING when calling the Answer Query Tool: {json.dumps(query_request)}. IMPORTANT: Make sure to properly JSON-encode the request_json parameter as a string when using the tool.",
            expected_output="JSON response with query results including answers and sources.",
            agent=query_agent,
        )

        result = query_agent.execute_task(task)

        try:
            if "```" in result:
                import re

                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result)
                if match:
                    result = match.group(1).strip()

            result_dict = json.loads(result)

            if "results" in result_dict and len(result_dict["results"]) > 0:
                query_result = result_dict["results"][0]
                return {
                    "answer": query_result.get("answer", ""),
                    "sources": query_result.get("sources", []),
                }
            return result_dict
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"answer": result, "sources": []}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}


st.title("📚 Document QA System")
st.markdown("""
Upload PDF, DOCX, or text files and ask questions about their content.
The system will process your documents and answer questions using RAG (Retrieval Augmented Generation).
""")

ingest_tool, query_tool, ingestion_agent, query_agent = initialize_tools_and_agents()

if (
    ingest_tool is None
    or query_tool is None
    or ingestion_agent is None
    or query_agent is None
):
    st.error("Failed to initialize tools and agents. Please check your API key.")
    st.stop()

tab1, tab2 = st.tabs(["Upload Documents", "Ask Questions"])

with tab1:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files to upload (PDF, DOCX, TXT, MD)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md"],
    )

    if uploaded_files and st.button("Process Documents", key="process_docs"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            current_progress = i / total_files
            progress_bar.progress(current_progress)
            status_text.text(
                f"Processing {uploaded_file.name}... ({i + 1}/{total_files})"
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            processing_message = st.empty()
            processing_message.info(f"Running ingestion on {uploaded_file.name}...")

            result = process_document(tmp_path, ingestion_agent)

            processing_message.empty()

            os.unlink(tmp_path)

            if result.get("status") == "completed":
                st.success(f"Successfully processed {uploaded_file.name}")
                st.json(result)
            else:
                st.error(f"Failed to process {uploaded_file.name}")
                st.json(result)

            after_progress = (i + 1) / total_files
            progress_bar.progress(after_progress)

        status_text.text(f"All {total_files} documents processed!")

with tab2:
    st.markdown('<p class="main-header">Ask Questions</p>', unsafe_allow_html=True)

    with st.form(key="query_form"):
        query = st.text_area("Ask a question about your documents:", height=100)
        submit_button = st.form_submit_button("Submit Question")

    results_container = st.container()

    if submit_button and query:
        st.session_state.user_query = query

        with st.spinner("Generating answer..."):
            if query_agent is not None:
                result = process_query(query, query_agent)
                st.session_state.query_result = result

    with results_container:
        if "query_result" in st.session_state and st.session_state.query_result:
            result = st.session_state.query_result

            if "answer" in result:
                if "user_query" in st.session_state and st.session_state.user_query:
                    st.markdown(
                        f'<div class="question-text">{st.session_state.user_query}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<p class="section-header">Answer</p>', unsafe_allow_html=True
                )
                st.markdown(result["answer"])

                if "sources" in result and result["sources"]:
                    st.markdown(
                        '<p class="section-header">Sources</p>', unsafe_allow_html=True
                    )
                    for i, source in enumerate(result["sources"]):
                        st.markdown(
                            f'<p class="source-header">Source {i + 1}: {source.get("document_type", "Document")} - {source.get("path", "Unknown path")}</p>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"**Content Type:** {source.get('content_type', 'text')}"
                        )

                        if source.get("page"):
                            st.markdown(f"**Page:** {source.get('page')}")

                        st.markdown("**Snippet:**")
                        st.markdown(source.get("snippet", "No snippet available"))

                        if source.get("content_type") == "image" and source.get(
                            "image_path"
                        ):
                            image_path = source.get("image_path")
                            st.markdown(f"**Image Path:** {image_path}")

                            try:
                                if os.path.exists(image_path):
                                    st.image(
                                        image_path,
                                        caption=f"Image from {source.get('path', 'document')}",
                                    )
                                else:
                                    st.warning(
                                        "Image file not found at the specified path."
                                    )
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")

                        st.markdown(
                            '<div class="custom-divider"></div>', unsafe_allow_html=True
                        )
            else:
                st.error("Failed to get answer")
                st.json(result)

st.sidebar.title("About")
st.sidebar.info("""
This Document QA System uses RAG (Retrieval Augmented Generation) to provide accurate answers from your documents.

1. Upload your documents in the first tab
2. Ask questions in the second tab
3. Get answers with source citations

Advanced RAG features enabled:
- HyDE (Hypothetical Document Embeddings): Improves retrieval by expanding queries
- Cross-encoder Reranking: Improves result relevance ordering
""")

if os.path.exists("db"):
    db_size = sum(f.stat().st_size for f in Path("db").glob("**/*") if f.is_file())
    db_size_mb = db_size / (1024 * 1024)
    st.sidebar.metric("Database Size", f"{db_size_mb:.2f} MB")

    if os.path.exists(os.path.join("db", "images")):
        image_count = len(list(Path(os.path.join("db", "images")).glob("**/*.png")))
        st.sidebar.metric("Stored Images", image_count)
