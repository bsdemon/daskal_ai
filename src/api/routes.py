from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from typing import Annotated, Optional, List
import io

from src.models.schemas import (
    DocumentBatch,
    DocumentIDs,
    Query,
    QueryResults,
    RAGRequest,
    RAGResponse,
)
from src.services.rag_service import RAGService
from src.utils.text_splitter import TextSplitter
from src.api.config_routes import config_router

# Create router
api_router = APIRouter()

# Include configuration routes
api_router.include_router(config_router)

# Create service instances
rag_service = RAGService()
text_splitter = TextSplitter()


@api_router.post("/documents", response_model=DocumentIDs)
async def add_documents(
    file: UploadFile = File(...),
    metadata_source: Optional[str] = Form(None)
):
    """Add document from file upload to the vector database.

    Similar to /upload endpoint but with a simpler interface for single document uploads.
    """
    try:
        # Read file content as bytes
        content = await file.read()

        # Extract text based on file type
        text = ""

        # Simple file type detection
        file_type = file.content_type if file.content_type else "text/plain"
        file_name = file.filename if file.filename else "unknown"

        # Try different decodings for text files
        if file_type.startswith("text/") or file_name.endswith((".txt", ".md", ".json", ".csv")):
            # Try different encodings, starting with UTF-8
            encodings = ["utf-8", "latin-1", "windows-1252", "ascii"]

            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if not text:
                # If all encodings fail, use latin-1 as fallback (it never fails)
                text = content.decode('latin-1')

        # For other file types, just use filename as text for now
        else:
            # Fallback for binary files - just use filename
            text = f"File: {file_name}"

        # Create metadata
        metadata = {
            "source": metadata_source or file_name,
            "filename": file_name,
            "content_type": file_type,
            "size_bytes": len(content)
        }

        # Add document to vector DB
        ids = await rag_service.add_documents([text], [metadata])
        return DocumentIDs(ids=ids)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )


@api_router.post("/documents/chunks", response_model=DocumentIDs)
async def add_chunked_documents(
    file: UploadFile = File(...),
    metadata_source: Optional[str] = Form(None)
):
    """Add document from file upload to the vector database after chunking it."""
    try:
        # Read file content as bytes
        content = await file.read()

        # Extract text based on file type
        text = ""

        # Simple file type detection
        file_type = file.content_type if file.content_type else "text/plain"
        file_name = file.filename if file.filename else "unknown"

        # Try different decodings for text files
        if file_type.startswith("text/") or file_name.endswith((".txt", ".md", ".json", ".csv")):
            # Try different encodings, starting with UTF-8
            encodings = ["utf-8", "latin-1", "windows-1252", "ascii"]

            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if not text:
                # If all encodings fail, use latin-1 as fallback (it never fails)
                text = content.decode('latin-1')

        # For other file types, just use filename as text for now
        else:
            # Fallback for binary files - just use filename
            text = f"File: {file_name}"

        # Create metadata
        metadata = {
            "source": metadata_source or file_name,
            "filename": file_name,
            "content_type": file_type,
            "size_bytes": len(content)
        }

        # Prepare document for chunking
        docs_for_chunking = [{"text": text, "metadata": metadata}]

        # Split document into chunks
        chunked_docs = text_splitter.split_documents(docs_for_chunking)

        # Extract text and metadata
        texts = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]

        # Add document chunks to vector DB
        ids = await rag_service.add_documents(texts, metadatas)
        return DocumentIDs(ids=ids)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process and chunk document: {str(e)}"
        )


@api_router.post("/documents/batch", response_model=DocumentIDs)
async def add_document_batch(documents: DocumentBatch):
    """Add multiple documents to the vector database using the batch schema.

    This endpoint maintains backward compatibility with clients using the DocumentBatch schema.
    """
    # Extract text and metadata
    texts = [doc.text for doc in documents.documents]
    metadatas = [doc.metadata for doc in documents.documents]

    # Add documents
    try:
        ids = await rag_service.add_documents(texts, metadatas)
        return DocumentIDs(ids=ids)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add documents: {str(e)}"
        )


@api_router.post("/documents/batch/chunks", response_model=DocumentIDs)
async def add_chunked_document_batch(documents: DocumentBatch):
    """Add documents to the vector database after chunking them using the batch schema.

    This endpoint maintains backward compatibility with clients using the DocumentBatch schema.
    """
    # Prepare documents for chunking
    docs_for_chunking = [
        {"text": doc.text, "metadata": doc.metadata} for doc in documents.documents
    ]

    # Split documents into chunks
    chunked_docs = text_splitter.split_documents(docs_for_chunking)

    # Extract text and metadata
    texts = [doc["text"] for doc in chunked_docs]
    metadatas = [doc["metadata"] for doc in chunked_docs]

    # Add documents
    try:
        ids = await rag_service.add_documents(texts, metadatas)
        return DocumentIDs(ids=ids)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add documents: {str(e)}"
        )


@api_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the vector database."""
    try:
        rag_service.vector_db.delete_document(doc_id)
        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@api_router.post("/search", response_model=QueryResults)
async def search_documents(query: Query):
    """Search for documents in the vector database."""
    try:
        results = await rag_service.search(
            query=query.text,
            n_results=query.n_results,
            where=query.where,
            rerank_method=query.rerank_method,
        )
        return QueryResults(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@api_router.post("/rag", response_model=RAGResponse)
async def generate_rag_response(request: RAGRequest):
    """Generate a response using Retrieval-Augmented Generation."""
    try:
        response = await rag_service.generate(
            query=request.query,
            n_results=request.n_results,
            where=request.where,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            llm_provider=request.llm_provider,
            embedding_provider=request.embedding_provider,
            rerank_method=request.rerank_method,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


@api_router.delete("/collection")
async def clear_collection():
    """Clear all documents from the collection."""
    try:
        rag_service.vector_db.clear_collection()
        return {"message": "Collection cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear collection: {str(e)}"
        )

@api_router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint for Kubernetes."""
    return {"status": "ok"}


@api_router.post("/upload", response_model=DocumentIDs)
async def upload_file(
    file: UploadFile = File(...),
    metadata_source: Optional[str] = Form(None),
    chunk: bool = Form(False)
):
    """Upload a file and add its content to the vector database.

    Handles binary files by attempting to extract text using appropriate methods
    based on file type. Supports various encodings beyond UTF-8.
    """
    try:
        # Read file content as bytes
        content = await file.read()

        # Extract text based on file type
        text = ""

        # Simple file type detection
        file_type = file.content_type if file.content_type else "text/plain"
        file_name = file.filename if file.filename else "unknown"

        # Try different decodings for text files
        if file_type.startswith("text/") or file_name.endswith((".txt", ".md", ".json", ".csv")):
            # Try different encodings, starting with UTF-8
            encodings = ["utf-8", "latin-1", "windows-1252", "ascii"]

            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if not text:
                # If all encodings fail, use latin-1 as fallback (it never fails)
                text = content.decode('latin-1')

        # For other file types, just use filename as text for now
        # In a production system, you would integrate with libraries like PyPDF2, docx, etc.
        else:
            # Fallback for binary files - just use filename
            text = f"File: {file_name}"

        # Create metadata
        metadata = {
            "source": metadata_source or file_name,
            "filename": file_name,
            "content_type": file_type,
            "size_bytes": len(content)
        }

        # Add document to vector DB, with chunking if requested
        if chunk:
            # Use the chunking endpoint logic
            docs_for_chunking = [{"text": text, "metadata": metadata}]
            chunked_docs = text_splitter.split_documents(docs_for_chunking)
            texts = [doc["text"] for doc in chunked_docs]
            metadatas = [doc["metadata"] for doc in chunked_docs]
            ids = await rag_service.add_documents(texts, metadatas)
        else:
            # Direct add
            ids = await rag_service.add_documents([text], [metadata])

        return DocumentIDs(ids=ids)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}"
        )