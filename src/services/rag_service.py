from typing import List, Dict, Any, Optional
from src.db.vector_db import ContextualVectorDB
from src.services.llm_factory import LLMFactory
from src.models.schemas import QueryResult, RAGResponse
from src.core.config import dynamic_settings as settings


class RAGService:
    """Service for Retrieval-Augmented Generation."""

    def __init__(
        self,
        collection_name: str = "contextual_documents",
        embedding_provider: str = None,
        llm_provider: str = None,
        rerank_method: str = None,
    ):
        self.vector_db = ContextualVectorDB(
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )
        self.default_rerank_method = rerank_method

    async def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to the vector database."""
        return await self.vector_db.add_documents(documents, metadatas)

    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: Dict[str, Any] = None,
        rerank_method: str = None,
    ) -> List[QueryResult]:
        """Search for documents relevant to the query."""
        method = rerank_method or self.default_rerank_method
        results = await self.vector_db.search(query, n_results, where, method)

        # Convert to QueryResult objects
        return [
            QueryResult(
                id=result["id"],
                text=result["metadata"].get("original_text", result["text"]),
                metadata=result["metadata"],
                score=result["score"],
            )
            for result in results
        ]

    def _format_rag_prompt(self, query: str, context: List[QueryResult]) -> str:
        """Format a RAG prompt with the query and context."""
        # Extract the original text from each result
        context_texts = []
        for i, result in enumerate(context):
            text = result.metadata.get("original_text", result.text)
            context_texts.append(f"[{i+1}] {text}")

        # Join all context texts
        all_context = "\n\n".join(context_texts)

        # Create the RAG prompt
        prompt = (
            f"Answer the following question based on the provided context. "
            f"If the context doesn't contain relevant information to answer the question, "
            f"acknowledge that and provide a general response.\n\n"
            f"Context:\n{all_context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return prompt

    async def generate(
        self,
        query: str,
        n_results: int = 10,
        where: Dict[str, Any] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        llm_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        rerank_method: Optional[str] = None,
    ) -> RAGResponse:
        """Generate a response to the query using RAG."""
        # Override the embedding provider if specified
        # Generate default system prompt if not provided
        rag_prompt = query
        results = None
        if system_prompt is None:
            system_prompt = settings.base_settings.SYSTEM_PROMPT
            
        if settings.base_settings.ENABLE_EMBEDDING:
            if (
                embedding_provider
                and embedding_provider != self.vector_db.embedding_client.__class__.__name__
            ):
                self.vector_db.embedding_client = embedding_provider

            # Search for relevant documents
            results = await self.search(query, n_results, where, rerank_method)

            rag_prompt = self._format_rag_prompt(query, results)
        
        # Create LLM client
        llm_client = LLMFactory.create_client(llm_provider)


        # Generate answer
        answer = await llm_client.generate(rag_prompt, system_prompt, temperature)
        print(f"Answer: {answer}")
        
        # Return response
        return RAGResponse(answer=answer, sources=results)
