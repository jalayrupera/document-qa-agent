# Document RAG System

An advanced document question-answering system using Retrieval-Augmented Generation (RAG) with CrewAI and EmbedChain.

## Features

- **Multi-Format Document Processing**: Supports PDF, DOCX, TXT, and MD files
- **Image Extraction & Description**: Automatically extracts images from PDFs and generates descriptions
- **Semantic Search**: Find relevant information across your document collection
- **Rich Metadata**: Maintains detailed information about document sources
- **Interactive Query Interface**: Natural language question answering with source citations

## Project Structure

```
.
├── main.py               # Main application script with interactive mode
├── tools/                # Core RAG components
│   ├── __init__.py       # Package initialization
│   ├── rag_ingest.py     # Document ingestion tool
│   └── rag_query.py      # Query answering tool
├── RAG_Plan.md           # Documentation of image extraction process
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (create from .env.example)
└── sample_docs/          # Directory for sample documents
```

## Setup

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your Google API key:

```
GEMINI_API_KEY=your_GEMINI_API_KEY_here
```

You can get a Google API key from https://makersuite.google.com/app/apikey.

## Usage

Run the interactive mode:

```bash
python main.py
```

The application provides two main functions:

### 1. Process Documents

When in "process" mode, you can add documents to the knowledge base:

- Enter the path to a document (PDF, DOCX, TXT, MD)
- The system will process and store the document with metadata
- Images from PDFs will be extracted, described, and stored

### 2. Query Documents

When in "query" mode, you can ask questions about the processed documents:

- Enter a natural language question
- The system will search for relevant information and generate an answer
- The answer will include citations to the source documents

## Technical Implementation

### Document Ingestion Process

1. **Document Loading**: Files are loaded based on their type (PDF, DOCX, TXT, MD)
2. **Text Extraction**: Content is extracted and split into appropriate chunks
3. **Image Processing**: For PDFs, images are extracted and converted to base64
4. **Description Generation**: Gemini 1.5 Flash generates descriptions for images
5. **Metadata Tagging**: Chunks are stored with detailed metadata
6. **Embedding Generation**: Text is vectorized for semantic search

### Query Process

1. **Query Processing**: User questions are analyzed
2. **Relevant Document Retrieval**: The system finds related chunks from the knowledge base
3. **Context Building**: Retrieved chunks are used to build a prompt context
4. **Answer Generation**: Gemini 1.5 Flash generates a comprehensive answer based on the context
5. **Source Citation**: The system provides citations to the original documents

## Dependencies

- crewai: Agent orchestration framework
- langchain: LLM integration
- langchain-google-genai: Gemini model integration
- embedchain: Vector database and embedding
- PyMuPDF: PDF processing
- python-docx: DOCX processing
- Pillow: Image processing
- pydantic: Data validation
- python-dotenv: Environment variable management
- chromadb: Vector database

## License

This project is licensed under the MIT License - see the LICENSE file for details. 