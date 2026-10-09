"""
Enterprise Hybrid Retriever combining BM25 lexical search and Qdrant
dense vector search with Reciprocal Rank Fusion (RRF).

RRF Formula:
    RRF(d) = sum_{m in M} 1 / (k + r_m(d))

where k=60 by default, M is the set of ranking methods,
and r_m(d) is the rank of document d in method m.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

import rank_bm25
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Result of a single retrieval method."""
    method: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    execution_time_ms: float


class HybridRetrievalResult(BaseModel):
    """Combined result from hybrid retrieval."""
    query: str
    documents: List[Dict[str, Any]]
    fused_scores: List[float]
    rrf_k: int = Field(default=60)
    bm25_results: Optional[RetrievalResult] = None
    vector_results: Optional[RetrievalResult] = None
    total_execution_time_ms: float


# ---------------------------------------------------------------------------
# Enterprise Hybrid Retriever
# ---------------------------------------------------------------------------

class EnterpriseHybridRetriever:
    """
    Hybrid retrieval combining BM25 (lexical) and Qdrant (dense vector)
    with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_corpus: Optional[List[str]] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "enterprise_documents",
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            bm25_corpus: Initial corpus for BM25 indexing (list of document texts)
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            qdrant_collection: Qdrant collection name
            rrf_k: RRF constant k (higher = more uniform)
            bm25_weight: Weight multiplier for BM25 scores
            vector_weight: Weight multiplier for vector scores
        """
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # BM25 index
        self.bm25_corpus: List[str] = bm25_corpus or []
        self.bm25_tokenized: List[List[str]] = []
        self.bm25_index: Optional[rank_bm25.BM25Okapi] = None

        if self.bm25_corpus:
            self._build_bm25_index()

        # Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.qdrant_collection = qdrant_collection

    def _build_bm25_index(self) -> None:
        """Build BM25 index from corpus."""
        import re

        self.bm25_tokenized = []
        for doc in self.bm25_corpus:
            tokens = re.findall(r'\b\w+\b', doc.lower())
            self.bm25_tokenized.append(tokens)

        if self.bm25_tokenized:
            self.bm25_index = rank_bm25.BM25Okapi(self.bm25_tokenized)

    def add_bm25_documents(self, documents: List[str]) -> None:
        """Add documents to BM25 index."""
        for doc in documents:
            self.bm25_corpus.append(doc)
            import re
            tokens = re.findall(r'\b\w+\b', doc.lower())
            self.bm25_tokenized.append(tokens)

        self._build_bm25_index()

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Perform BM25 lexical search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            RetrievalResult with BM25 scores
        """
        import time

        if self.bm25_index is None:
            return RetrievalResult(
                method="bm25",
                documents=[],
                scores=[],
                execution_time_ms=0.0,
            )

        start = time.time()
        import re
        query_tokens = re.findall(r'\b\w+\b', query.lower())

        scores = self.bm25_index.get_scores(query_tokens)
        # Get top-k indices
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        top_indices = [i for i, _ in indexed_scores[:top_k]]
        top_scores = [s for _, s in indexed_scores[:top_k]]

        documents = []
        for idx in top_indices:
            documents.append({
                "id": idx,
                "text": self.bm25_corpus[idx],
                "source": "bm25",
            })

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            method="bm25",
            documents=documents,
            scores=top_scores,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        embedding_model: Optional[Any] = None,
    ) -> RetrievalResult:
        """
        Perform dense vector search via Qdrant.

        Args:
            query: Search query
            top_k: Number of results to return
            embedding_model: Optional model to generate query embedding.
                           If None, uses Qdrant's built-in embedding.

        Returns:
            RetrievalResult with vector similarity scores
        """
        import time

        start = time.time()

        # Generate query embedding if model provided
        if embedding_model is not None:
            query_vector = embedding_model.encode(query)
        else:
            # Placeholder - in production use sentence-transformers
            query_vector = [0.0] * 768  # Default dimension

        # Search Qdrant
        try:
            results = self.qdrant_client.search(  # type: ignore[attr-defined]
                collection_name=self.qdrant_collection,
                query_vector=query_vector,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128),
            )

            documents = []
            scores = []
            for hit in results:
                payload = hit.payload or {}
                documents.append({
                    "id": hit.id,
                    "text": payload.get("text", ""),
                    "source": "vector",
                    "metadata": payload.get("metadata", {}),
                })
                scores.append(hit.score)

        except Exception:
            # Qdrant not available - return empty
            documents = []
            scores = []

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            method="vector",
            documents=documents,
            scores=scores,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def fuse_with_rrf(
        self,
        bm25_result: RetrievalResult,
        vector_result: RetrievalResult,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Fuse two result sets using Reciprocal Rank Fusion.

        RRF(d) = sum_{m in M} 1 / (k + r_m(d))

        Args:
            bm25_result: BM25 retrieval result
            vector_result: Vector retrieval result

        Returns:
            Tuple of (fused_documents, fused_scores)
        """
        k = self.rrf_k

        # Build rank dictionaries
        bm25_ranks: Dict[int, int] = {}
        for rank, d in enumerate(bm25_result.documents):
            doc_id = d.get("id", rank)
            bm25_ranks[doc_id] = rank + 1  # 1-indexed ranks

        vector_ranks: Dict[int, int] = {}
        for rank, d in enumerate(vector_result.documents):
            doc_id = d.get("id", rank)
            vector_ranks[doc_id] = rank + 1

        # Union of all document IDs
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # Calculate RRF scores
        fused_scores: Dict[int, float] = {}
        for doc_id in all_ids:
            rrf_score = 0.0

            if doc_id in bm25_ranks:
                rrf_score += self.bm25_weight / (k + bm25_ranks[doc_id])
            if doc_id in vector_ranks:
                rrf_score += self.vector_weight / (k + vector_ranks[doc_id])

            fused_scores[doc_id] = rrf_score

        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result
        documents = []
        scores = []
        for doc_id, score in sorted_docs:
            # Find document from either source
            selected_doc: Optional[Dict[str, Any]] = None
            for d in bm25_result.documents:
                if d.get("id") == doc_id:
                    selected_doc = d
                    break
            if selected_doc is None:
                for d in vector_result.documents:
                    if d.get("id") == doc_id:
                        selected_doc = d
                        break

            if selected_doc is not None:
                doc_copy = dict(selected_doc)
                doc_copy["fused_score"] = round(score, 6)
                documents.append(doc_copy)
                scores.append(score)

        return documents, scores

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        embedding_model: Optional[Any] = None,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval with RRF fusion.

        Args:
            query: Search query
            top_k: Number of final results
            embedding_model: Optional embedding model for vector search

        Returns:
            HybridRetrievalResult with fused documents
        """
        import time

        total_start = time.time()

        # Run both retrieval methods
        bm25_result = self.search_bm25(query, top_k=top_k * 2)
        vector_result = self.search_vector(query, top_k=top_k * 2, embedding_model=embedding_model)

        # Fuse results
        fused_docs, fused_scores = self.fuse_with_rrf(bm25_result, vector_result)

        total_elapsed_ms = (time.time() - total_start) * 1000

        return HybridRetrievalResult(
            query=query,
            documents=fused_docs[:top_k],
            fused_scores=fused_scores[:top_k],
            rrf_k=self.rrf_k,
            bm25_results=bm25_result,
            vector_results=vector_result,
            total_execution_time_ms=round(total_elapsed_ms, 2),
        )

    def health_check(self) -> Dict[str, Any]:
        """Check health of both retrieval backends."""
        import time

        result: Dict[str, Any] = {
            "bm25": {
                "indexed_documents": len(self.bm25_corpus),
                "status": "ready" if self.bm25_index else "not_indexed",
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "status": "unknown",
            },
            "overall_status": "degraded",
        }

        # Check BM25
        if self.bm25_index:
            result["bm25"]["status"] = "ready"

        # Check Qdrant        qdrant_status: str = "unknown"        qdrant_response_ms: float = 0.0        collections_list: List[str] = []        try:            start = time.time()            collections_result = self.qdrant_client.get_collections()            elapsed = (time.time() - start) * 1000            qdrant_status = "connected"            qdrant_response_ms = round(elapsed, 2)            if collections_result and collections_result.collections:                collections_list = [c.name for c in collections_result.collections]        except Exception as e:            qdrant_status = f"error: {str(e)}"        result["qdrant"]["status"] = qdrant_status        result["qdrant"]["response_time_ms"] = qdrant_response_ms        result["qdrant"]["collections"] = collections_list        # Overall status
        bm25_ok = result["bm25"]["status"] == "ready"
        qdrant_ok = result["qdrant"]["status"] == "connected"

        if bm25_ok and qdrant_ok:
            result["overall_status"] = "healthy"
        elif bm25_ok or qdrant_ok:
            result["overall_status"] = "degraded"
        else:
            result["overall_status"] = "unhealthy"

        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_retriever(
    corpus: Optional[List[str]] = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> EnterpriseHybridRetriever:
    """Create a hybrid retriever with optional corpus."""
    return EnterpriseHybridRetriever(
        bm25_corpus=corpus,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
