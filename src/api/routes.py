from fastapi import APIRouter, HTTPException, status

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
async def add_documents(documents: DocumentBatch):
    """Add multiple documents to the vector database."""
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


@api_router.post("/documents/chunks", response_model=DocumentIDs)
async def add_chunked_documents(documents: DocumentBatch):
    """Add documents to the vector database after chunking them."""
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