from typing import List, Dict, Any
import uuid
import os
import json
import chromadb
from src.core.config import dynamic_settings as settings
from src.services.embedding_factory import EmbeddingFactory
from src.services.llm_factory import LLMFactory
from src.services.reranking_factory import RerankerFactory


class VectorDB:
    """Base vector database class using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_provider: str = None,
    ):
        # Initialize ChromaDB client
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding client if embeddings are enabled
        self.embedding_enabled = settings.ENABLE_EMBEDDING
        self.embedding_client = None
        if self.embedding_enabled:
            self.embedding_client = EmbeddingFactory.create_client(embedding_provider)

    async def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to the vector database."""
        if not documents:
            return []

        # Generate IDs if not provided in metadata
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]

        # Generate unique IDs for each document
        ids = [str(uuid.uuid4()) for _ in documents]

        # Generate embeddings if enabled, otherwise use None
        embeddings = None
        if self.embedding_enabled and self.embedding_client:
            embeddings = await self.embedding_client.embed(documents)

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
        )

        return ids

    async def search(
        self, query: str, n_results: int = 10, where: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity or keyword search."""
        # If embedding is enabled, use vector search
        if self.embedding_enabled and self.embedding_client:
            # Generate query embedding
            query_embedding = await self.embedding_client.embed([query])

            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding, n_results=n_results, where=where
            )
        else:
            # Fallback to keyword search
            results = self.collection.query(
                query_texts=[query], n_results=n_results, where=where
            )

        # Format results
        formatted_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": doc,
                        "metadata": results["metadatas"][0][i],
                        "score": (
                            float(results["distances"][0][i])
                            if "distances" in results
                            else None
                        ),
                    }
                )

        return formatted_results

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the vector database."""
        self.collection.delete(ids=[doc_id])

    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.collection.delete(where={})


class ContextualVectorDB(VectorDB):
    """Enhanced vector database that adds context to document chunks before embedding."""

    def __init__(
        self,
        collection_name: str = "contextual_documents",
        embedding_provider: str = None,
        llm_provider: str = None,
        cache_file: str = "context_cache.json",
    ):
        super().__init__(collection_name, embedding_provider)

        # Initialize contextual embedding settings
        self.contextual_embedding_enabled = (
            settings.ENABLE_CONTEXTUAL_EMBEDDING and settings.ENABLE_EMBEDDING
        )
        self.reranking_enabled = settings.ENABLE_RERANKING

        # Initialize LLM client if contextual embedding is enabled
        self.llm_client = None
        if self.contextual_embedding_enabled:
            self.llm_client = LLMFactory.create_client(llm_provider)

        # Setup context cache
        self.cache_file = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, cache_file)
        self.context_cache = (
            self._load_cache() if self.contextual_embedding_enabled else {}
        )

    def _load_cache(self) -> Dict[str, str]:
        """Load context cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except: # noqa: E722
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save context cache to file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.context_cache, f)

    async def _generate_context(self, text: str) -> str:
        """Generate contextual description for a text chunk."""
        # If contextual embedding is disabled, return empty string
        if not self.contextual_embedding_enabled or not self.llm_client:
            return ""

        # Check if context is already in cache
        text_hash = str(hash(text))
        if text_hash in self.context_cache:
            return self.context_cache[text_hash]

        # Generate context using LLM
        prompt = (
            "Generate a concise, factual description of what this text is about. "
            "Focus on key topics, entities, and concepts. Be specific but brief.\n\n"
            f"Text: {text}\n\n"
            "Description:"
        )

        context = await self.llm_client.generate(prompt)

        # Clean and store in cache
        context = context.strip()
        self.context_cache[text_hash] = context
        self._save_cache()

        return context

    async def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents with contextual descriptions to the vector database."""
        if not documents:
            return []

        # Generate IDs and prepare metadata
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]

        ids = [str(uuid.uuid4()) for _ in documents]

        # If contextual embedding is enabled, generate enhanced docs
        if (
            self.contextual_embedding_enabled
            and self.embedding_enabled
            and self.llm_client
        ):
            enhanced_docs = []
            enhanced_metadatas = []

            # Generate contextual descriptions for each document
            for i, doc in enumerate(documents):
                context = await self._generate_context(doc)

                # Create enhanced document with context
                enhanced_doc = (
                    f"Context: {context}\n\nContent: {doc}" if context else doc
                )
                enhanced_docs.append(enhanced_doc)

                # Add context to metadata
                meta = metadatas[i].copy()
                if context:
                    meta["context"] = context
                meta["original_text"] = doc
                enhanced_metadatas.append(meta)

            # Generate embeddings for enhanced documents if embedding is enabled
            embeddings = None
            if self.embedding_enabled and self.embedding_client:
                embeddings = await self.embedding_client.embed(enhanced_docs)

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=enhanced_docs,
                metadatas=enhanced_metadatas,
                ids=ids,
            )
        else:
            # Just add documents normally without contextual enhancement
            return await super().add_documents(documents, metadatas)

        return ids

    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: Dict[str, Any] = None,
        rerank_method: str = None,
    ) -> List[Dict[str, Any]]:
        """Search with enhanced contextual query and optional reranking."""
        # First get standard vector search results
        results = await super().search(query, n_results, where)

        # Apply reranking if enabled and specified
        if self.reranking_enabled and rerank_method and results:
            # Extract original texts from metadata for reranking
            original_texts = [
                result["metadata"].get("original_text", result["text"])
                for result in results
            ]
            scores = [result.get("score", 0.0) for result in results]

            # Create reranker and apply
            try:
                reranker = RerankerFactory.create_client(rerank_method)
                reranked = await reranker.rerank(query, original_texts, scores)

                # Reorganize results based on reranking
                reranked_results = []
                for doc, score in reranked:
                    # Find the original result for this document
                    for result in results:
                        if (
                            result["metadata"].get("original_text", result["text"])
                            == doc
                        ):
                            result_copy = result.copy()
                            result_copy["score"] = score
                            reranked_results.append(result_copy)
                            break

                return reranked_results
            except Exception as e:
                # If reranking fails, just return the original results
                print(f"Reranking error: {str(e)}")

        return results
