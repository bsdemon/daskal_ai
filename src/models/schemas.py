from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Schema for a document to be added to the vector database."""
    text: str = Field(..., description="The text content of the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata for the document")

class DocumentBatch(BaseModel):
    """Schema for a batch of documents."""
    documents: List[Document] = Field(..., description="List of documents to add")

class DocumentID(BaseModel):
    """Schema for document identifiers."""
    id: str = Field(..., description="Document identifier")

class DocumentIDs(BaseModel):
    """Schema for a list of document identifiers."""
    ids: List[str] = Field(..., description="List of document identifiers")

class Query(BaseModel):
    """Schema for a query to the vector database."""
    text: str = Field(..., description="The query text")
    n_results: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    where: Optional[Dict[str, Any]] = Field(default=None, description="Filter criteria")
    llm_provider: Optional[str] = Field(default=None, description="LLM provider to use")
    embedding_provider: Optional[str] = Field(default=None, description="Embedding provider to use")
    rerank_method: Optional[str] = Field(default=None, description="Reranking method to use")
    
class QueryResult(BaseModel):
    """Schema for a query result."""
    id: str = Field(..., description="Document identifier")
    text: str = Field(..., description="Document text")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: Optional[float] = Field(default=None, description="Relevance score")

class QueryResults(BaseModel):
    """Schema for query results."""
    results: List[QueryResult] = Field(..., description="List of query results")

class RAGRequest(BaseModel):
    """Schema for a RAG request."""
    query: str = Field(..., description="The user query")
    n_results: int = Field(default=10, description="Number of results to retrieve", ge=1, le=100)
    where: Optional[Dict[str, Any]] = Field(default=None, description="Filter criteria")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the LLM")
    temperature: Optional[float] = Field(default=None, description="Temperature for the LLM", ge=0.0, le=1.0)
    llm_provider: Optional[str] = Field(default=None, description="LLM provider to use")
    embedding_provider: Optional[str] = Field(default=None, description="Embedding provider to use")
    rerank_method: Optional[str] = Field(default=None, description="Reranking method to use")

class RAGResponse(BaseModel):
    """Schema for a RAG response."""
    answer: str = Field(..., description="The generated answer")
    sources: List[QueryResult] = Field(..., description="The sources used for the answer")

class HealthResponse(BaseModel):
    """Simple health check response for Kubernetes."""
    status: str = Field(..., description="API health status")