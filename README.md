# Document QA Agent

A Retrieval-Augmented Generation (RAG) system for querying documents with natural language questions. This system processes documents (PDF, DOCX, TXT, MD) from local uploads or URLs, extracts text and images, and enables semantic search and question answering based on the document content.

## Features

- Document ingestion with support for PDF, DOCX, TXT, and Markdown files
- URL-based document ingestion for remote documents
- Image extraction and analysis from documents
- Text chunking and embedding for efficient retrieval
- Natural language querying with source citations
- Web interface built with Streamlit
- Docker support for easy deployment

## Requirements

- Python 3.10+
- Google Gemini API key

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-qa-agent.git
   cd document-qa-agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Docker Setup

1. Make sure Docker and Docker Compose are installed on your system.

2. Create a `.env` file with your API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Build and start the container:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:8501

5. To stop the application:
   ```bash
   docker-compose down
   ```

## Usage

1. **Upload Documents**: 
   - Go to the "Upload Documents" tab
   - **Local Files**: Select one or more documents to upload (PDF, DOCX, TXT, MD)
   - **URL Documents**: Enter the URL of a document (PDF, DOCX, TXT, MD) and click "Add URL"
   - You can add multiple URLs and remove them individually or clear all
   - Click "Process Documents" to ingest them into the knowledge base

2. **Ask Questions**:
   - Go to the "Ask Questions" tab
   - Enter your question about the document content
   - Click "Submit Question" to get an answer with source citations

## Data Storage

- Document chunks and embeddings are stored in a local ChromaDB database in the `db` directory
- Extracted images are saved to `db/images` with a structured naming convention

## License

This project is licensed under the MIT License - see the LICENSE file for details. 