"""PDF document loader and processor."""

from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Backward compatibility with older LangChain versions.
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from ..config import settings


class PDFLoader:
    """Load and process PDF documents."""

    def __init__(self):
        self.documents_folder = settings.documents_folder
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        # Ensure documents folder exists
        self.documents_folder.mkdir(parents=True, exist_ok=True)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\nArtículo ",
                "\nDisposición ",
                "\nCapítulo ",
                "\nTítulo ",
                "\nAnexo ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

    def load_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        Load a single PDF and split into chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of document chunks with metadata
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)

            # Format chunks with metadata
            formatted_chunks = []
            for i, chunk in enumerate(chunks):
                formatted_chunks.append(
                    {
                        "content": chunk.page_content,
                        "metadata": {
                            "filename": pdf_path.name,
                            "page": chunk.metadata.get("page", 0),
                            "chunk_id": i,
                            "source": str(pdf_path),
                        },
                    }
                )

            return formatted_chunks

        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            return []

    def load_all_pdfs(self) -> List[Dict[str, str]]:
        """
        Load all PDFs from documents folder.

        Returns:
            List of all document chunks from all PDFs
        """
        all_chunks = []

        pdf_files = list(self.documents_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.documents_folder}")
            return []

        print(f"Loading {len(pdf_files)} PDF files...")

        for pdf_path in pdf_files:
            chunks = self.load_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {pdf_path.name}")

        print(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks

    def get_pdf_list(self) -> List[str]:
        """
        Get list of PDF filenames in documents folder.

        Returns:
            List of PDF filenames
        """
        return [pdf.name for pdf in self.documents_folder.glob("*.pdf")]
