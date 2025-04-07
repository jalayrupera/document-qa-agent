import base64
import json
import os
from io import BytesIO
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import docx
import fitz
import google.genai as genai
from crewai.tools import BaseTool
from embedchain import App
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    paths: List[str] = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(None)


class IngestResponse(BaseModel):
    status: str = Field(...)
    processed_files: List[str] = Field(...)
    document_ids: List[str] = Field(...)
    chunks_count: int = Field(...)
    errors: Optional[Dict[str, str]] = Field(None)


class IngestDataTool(BaseTool):
    name: str = Field(default="Ingest Data Tool")
    description: str = Field(default="Tool to ingest documents into the knowledge base")
    
    model_config = {"arbitrary_types_allowed": True}

    CHUNK_SIZE: ClassVar[int] = 10000
    CHUNK_OVERLAP: ClassVar[int] = 100
    text_splitter: Optional[RecursiveCharacterTextSplitter] = Field(default=None)

    def __init__(self):
        super().__init__(name="Ingest Data Tool", description="Tool to ingest documents into the knowledge base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def _init_gemini_llm(self):
        """Initializes the Google Generative AI client."""
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

    def _save_image_to_folder(self, image: Image.Image, document_id: str, page_num: int, img_index: int) -> str:
        """Save image to a folder and return the path."""
        # Create images directory if it doesn't exist
        images_dir = os.path.join("db", "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create document-specific subfolder
        doc_images_dir = os.path.join(images_dir, document_id)
        os.makedirs(doc_images_dir, exist_ok=True)
        
        # Generate filename with page number and image index
        filename = f"page_{page_num+1}_img_{img_index}.png"
        image_path = os.path.join(doc_images_dir, filename)
        
        # Save image
        image.save(image_path)
        
        return image_path

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def _validate_file_path(self, path: str) -> bool:
        if not os.path.exists(path):
            return False

        if not os.path.isfile(path):
            return False

        return True

    def _extract_text_from_page(self, page):
        try:
            text = str(page)
            if text.strip():
                return text
        except Exception:
            pass

        try:
            blocks = page.get_text("blocks")
            if blocks:
                return "\n".join(
                    [block[4] for block in blocks if isinstance(block[4], str)]
                )
        except Exception:
            pass

        try:
            words = page.get_text("words")
            if words:
                return " ".join([word[4] for word in words if isinstance(word[4], str)])
        except Exception:
            pass

        return ""

    def _convert_page_to_image(self, page, zoom=2.0):
        """Convert a PDF page to a high-resolution image."""
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        return Image.open(BytesIO(img_data))

    def _perform_ocr_with_gemini(self, image: Image.Image, client) -> str:
        """Perform OCR on an image using Gemini and return the extracted text."""
        try:
            prompt = """
            Please extract all text from this image. Include:
            1. All visible text in the image
            2. Maintain the original formatting and layout as much as possible
            3. Include any headers, footers, or marginal text
            4. Preserve any special characters or symbols
            
            Return ONLY the extracted text, without any additional commentary or description.
            """
            
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[prompt, image]
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            print(f"Error performing OCR with Gemini: {e}")
            return ""

    def _process_pdf(
        self, file_path: str, metadata: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        ec_app = self._get_embedchain_app()
        document_chunks = []
        document_id = os.path.basename(file_path)
        client = self._init_gemini_llm()

        try:
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)

            for page_num in range(total_pages):
                page = pdf_document[page_num]
                
                # Convert page to image
                page_image = self._convert_page_to_image(page)
                
                # Save the page image
                page_image_path = self._save_image_to_folder(page_image, document_id, page_num, 0)
                
                # Perform OCR using Gemini
                ocr_text = self._perform_ocr_with_gemini(page_image, client)
                
                # Get regular text extraction as backup
                regular_text = self._extract_text_from_page(page)
                
                # Combine OCR text and regular text
                combined_text = f"{ocr_text}\n{regular_text}".strip()
                
                # Create mandatory metadata
                page_metadata = {
                    "document_id": document_id,
                    "pdf_path": file_path,
                    "page_number": page_num + 1,
                    "page_image": page_image_path,
                    "content_type": "text",
                    "document_type": "pdf",
                    "total_pages": total_pages,
                    "has_ocr": bool(ocr_text.strip()),
                    "has_regular_text": bool(regular_text.strip()),
                    "source": file_path,
                    "source_type": "pdf",
                    "source_page": page_num + 1,
                    # Add these additional source-related fields for embedchain
                    "url": file_path,
                    "title": f"{os.path.basename(file_path)} - Page {page_num + 1}",
                    "source_id": f"{document_id}_page_{page_num + 1}",
                    "chunk_id": f"{document_id}_page_{page_num + 1}_text"
                }
                
                # Add any additional metadata from input
                if metadata:
                    page_metadata.update(metadata)

                if combined_text:
                    ec_app.add(combined_text, metadata=page_metadata)
                    document_chunks.append(page_metadata)

                # Process embedded images in the page
                images = page.get_images(full=True)
                for img_index, img_info in enumerate(images):
                    try:
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(BytesIO(image_bytes))

                        if image.width < 50 or image.height < 50:
                            continue

                        # Save embedded image
                        image_path = self._save_image_to_folder(image, document_id, page_num, img_index + 1)

                        # Create mandatory metadata for embedded image
                        image_metadata = {
                            "document_id": document_id,
                            "pdf_path": file_path,
                            "page_number": page_num + 1,
                            "page_image": page_image_path,
                            "embedded_image": image_path,
                            "content_type": "image",
                            "document_type": "pdf",
                            "total_pages": total_pages,
                            "image_index": img_index,
                            "image_width": image.width,
                            "image_height": image.height,
                            "is_embedded": True,
                            "source": file_path,
                            "source_type": "pdf",
                            "source_page": page_num + 1,
                            "source_image_index": img_index,
                            # Add these additional source-related fields for embedchain
                            "url": file_path,
                            "title": f"{os.path.basename(file_path)} - Page {page_num + 1} - Image {img_index + 1}",
                            "source_id": f"{document_id}_page_{page_num + 1}_image_{img_index + 1}",
                            "chunk_id": f"{document_id}_page_{page_num + 1}_image_{img_index + 1}"
                        }
                        
                        # Add any additional metadata from input
                        if metadata:
                            image_metadata.update(metadata)

                        prompt = """
                        Describe the image in detail. Include:
                        1. What is shown in the image
                        2. Any text visible in the image
                        3. The purpose or context of the image
                        
                        Provide a comprehensive description that would help someone understand what's in the image without seeing it.
                        """

                        try:
                            response = client.models.generate_content(
                                model='gemini-2.0-flash',
                                contents=[prompt, image]
                            )
                            description = response.text if response.text else ""
                            description = description.strip()

                            if description and len(description) > 10:
                                ec_app.add(description, metadata=image_metadata)
                                document_chunks.append(image_metadata)
                        except Exception as e:
                            print(f"Error generating image description: {e}")
                            pass
                    except Exception as e:
                        print(f"Error processing embedded image {img_index}: {e}")
                        pass

            return document_id, document_chunks

        except Exception as e:
            raise ValueError(f"Error processing PDF {file_path}: {str(e)}")

    def _process_docx(
        self, file_path: str, metadata: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        ec_app = self._get_embedchain_app()

        document_chunks = []
        document_id = os.path.basename(file_path)

        try:
            doc = docx.Document(file_path)

            full_text = "\n".join(
                [para.text for para in doc.paragraphs if para.text.strip()]
            )

            if not full_text.strip():
                return document_id, document_chunks

            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update(
                {
                    "document_id": document_id,
                    "path": file_path,
                    "document_type": "docx",
                    "content_type": "text",
                }
            )

            ec_app.add(full_text, metadata=doc_metadata)
            document_chunks.append(doc_metadata)

            return document_id, document_chunks

        except Exception as e:
            raise ValueError(f"Error processing DOCX {file_path}: {str(e)}")

    def _process_text_file(
        self, file_path: str, metadata: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        ec_app = self._get_embedchain_app()

        document_chunks = []
        document_id = os.path.basename(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                return document_id, document_chunks

            file_extension = os.path.splitext(file_path)[1].lower()
            file_type = file_extension.replace(".", "") if file_extension else "txt"

            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update(
                {
                    "document_id": document_id,
                    "path": file_path,
                    "document_type": file_type,
                    "content_type": "text",
                }
            )

            ec_app.add(text, metadata=doc_metadata)
            document_chunks.append(doc_metadata)

            return document_id, document_chunks

        except Exception as e:
            raise ValueError(f"Error processing text file {file_path}: {str(e)}")

    def _process_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not self._validate_file_path(file_path):
            raise ValueError(f"Invalid file path: {file_path}")

        metadata = metadata or {}
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            return self._process_pdf(file_path, metadata)
        elif file_extension == ".docx":
            return self._process_docx(file_path, metadata)
        else:
            return self._process_text_file(file_path, metadata)

    def _run(self, request_json: str) -> str:
        try:
            request_data = json.loads(request_json)
            request = IngestRequest(**request_data)

            processed_files = []
            document_ids = []
            errors = {}
            total_chunks = 0

            for path in request.paths:
                try:
                    document_id, chunks = self._process_file(path, request.metadata)

                    processed_files.append(path)
                    document_ids.append(document_id)
                    total_chunks += len(chunks)

                except Exception as file_error:
                    errors[path] = str(file_error)

            response = IngestResponse(
                status="completed",
                processed_files=processed_files,
                document_ids=document_ids,
                chunks_count=total_chunks,
                errors=errors if errors else None,
            )

            return json.dumps(response.model_dump())

        except Exception as e:
            error_response = IngestResponse(
                status="failed",
                processed_files=[],
                document_ids=[],
                chunks_count=0,
                errors={"general_error": str(e)},
            )

            return json.dumps(error_response.model_dump())
