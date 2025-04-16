from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import re
import math
from collections import Counter
from src.core.config import dynamic_settings as settings


class RerankerClient(ABC):
    """Abstract base class for reranking clients."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[str], scores: List[float] = None
    ) -> List[Tuple[str, float]]:
        """Rerank the documents based on the query."""
        pass


class BM25Reranker(RerankerClient):
    """BM25 reranking implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenizer = lambda text: re.findall(r"\w+", text.lower())

    def _compute_tf(self, doc: str) -> Dict[str, int]:
        """Compute term frequency for a document."""
        tokens = self.tokenizer(doc)
        return Counter(tokens)

    def _compute_idf(
        self, query_tokens: List[str], documents: List[str]
    ) -> Dict[str, float]:
        """Compute inverse document frequency for query terms."""
        N = len(documents)
        idf = {}

        for token in query_tokens:
            df = sum(1 for doc in documents if token in self.tokenizer(doc))
            idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return idf

    async def rerank(
        self, query: str, documents: List[str], scores: List[float] = None
    ) -> List[Tuple[str, float]]:
        """Rerank documents using BM25 algorithm."""
        query_tokens = self.tokenizer(query)

        if not query_tokens or not documents:
            return [(doc, 0.0) for doc in documents]

        idf = self._compute_idf(query_tokens, documents)
        tf_docs = [self._compute_tf(doc) for doc in documents]

        # Calculate average document length
        avg_doc_len = sum(len(self.tokenizer(doc)) for doc in documents) / len(
            documents
        )

        # Calculate BM25 scores
        bm25_scores = []
        for i, doc in enumerate(documents):
            score = 0.0
            doc_len = len(self.tokenizer(doc))
            tf = tf_docs[i]

            for token in query_tokens:
                if token not in tf:
                    continue

                numerator = tf[token] * (self.k1 + 1)
                denominator = tf[token] + self.k1 * (
                    1 - self.b + self.b * doc_len / avg_doc_len
                )
                score += idf.get(token, 0) * numerator / denominator

            # Combine with original score if provided
            if scores:
                score = score * 0.7 + scores[i] * 0.3

            bm25_scores.append((doc, score))

        # Sort by score in descending order
        return sorted(bm25_scores, key=lambda x: x[1], reverse=True)


class CohereReranker(RerankerClient):
    """Reranker using Cohere's reranking API."""

    def __init__(self):
        if not settings.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not set")
        import cohere

        self.client = cohere.Client(settings.COHERE_API_KEY)

    async def rerank(
        self, query: str, documents: List[str], scores: List[float] = None
    ) -> List[Tuple[str, float]]:
        """Rerank documents using Cohere's reranking model."""
        if not documents:
            return []

        response = self.client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=len(documents),
        )

        # Extract results and convert to the expected format
        reranked_docs = [
            (documents[result.index], result.relevance_score)
            for result in response.results
        ]
        return reranked_docs


class RerankerFactory:
    """Factory for creating reranking clients."""

    @staticmethod
    def create_client(method: str = None) -> RerankerClient:
        """Create a reranker client based on the specified method."""
        method = method or settings.DEFAULT_RERANKING_METHOD

        if method == "bm25":
            return BM25Reranker()
        elif method == "cohere":
            return CohereReranker()
        else:
            raise ValueError(f"Unknown reranking method: {method}")
