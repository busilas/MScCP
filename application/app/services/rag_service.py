"""
RAG Service

Retrieval-Augmented Generation using FAISS for document retrieval.
"""
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np


class RAGService:
    """
    RAG service using FAISS for document retrieval.

    Manages document indexing and similarity search
    for retrieval-augmented generation.
    """

    def __init__(self, index_path: str = "data/faiss_index", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG service.

        Args:
            index_path: Path to store FAISS index
            embedding_model: Model for generating embeddings
        """
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        self.index = None
        self.documents = []
        self.document_metadata = []

        # Create index directory
        os.makedirs(index_path, exist_ok=True)

        # Load embedding model
        self._load_embedding_model()

        # Load existing index if available
        self._load_index()

    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"✓ Embedding model loaded: {self.embedding_model_name}")
        except ImportError:
            print("⚠ sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.embedding_model = None
        except Exception as e:
            print(f"⚠ Failed to load embedding model: {e}")
            self.embedding_model = None

    def _load_index(self):
        """Load existing FAISS index from disk."""
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")

        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                import faiss
                self.index = faiss.read_index(index_file)

                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)

                if os.path.exists(metadata_file):
                    with open(metadata_file, 'rb') as f:
                        self.document_metadata = pickle.load(f)

                print(f"✓ FAISS index loaded: {len(self.documents)} documents")
            except ImportError:
                print("⚠ faiss not installed. Install with: pip install faiss-cpu")
            except Exception as e:
                print(f"⚠ Failed to load FAISS index: {e}")

    def _save_index(self):
        """Save FAISS index to disk."""
        if self.index is None:
            return

        try:
            import faiss
            index_file = os.path.join(self.index_path, "index.faiss")
            docs_file = os.path.join(self.index_path, "documents.pkl")
            metadata_file = os.path.join(self.index_path, "metadata.pkl")

            faiss.write_index(self.index, index_file)

            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)

            with open(metadata_file, 'wb') as f:
                pickle.dump(self.document_metadata, f)

            print(f"✓ FAISS index saved: {len(self.documents)} documents")
        except Exception as e:
            print(f"⚠ Failed to save FAISS index: {e}")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add document to FAISS index.

        Args:
            text: Document text
            metadata: Optional metadata (filename, source, etc.)

        Returns:
            Number of chunks added
        """
        if self.embedding_model is None:
            print("⚠ Embedding model not available, using text-only storage")
            # Still chunk and store documents even without embeddings
            chunks = self.chunk_text(text)
            for chunk in chunks:
                self.documents.append(chunk)
                self.document_metadata.append(metadata or {})
            self._save_index()
            print(f"✓ Added {len(chunks)} text chunks (no embeddings)")
            return len(chunks)

        try:
            import faiss

            # Chunk document
            chunks = self.chunk_text(text)

            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks)

            # Initialize index if needed
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)

            # Add to index
            self.index.add(np.array(embeddings).astype('float32'))

            # Store documents and metadata
            for chunk in chunks:
                self.documents.append(chunk)
                self.document_metadata.append(metadata or {})

            # Save index
            self._save_index()

            return len(chunks)
        except Exception as e:
            print(f"Error adding document: {e}")
            return 0

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with text and similarity scores
        """
        # Fallback to keyword search if no embeddings
        if self.embedding_model is None or self.index is None:
            return self._keyword_search(query, top_k)

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Search index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'),
                min(top_k, len(self.documents))
            )

            # Format results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    # Convert L2 distance to similarity score (0-1)
                    similarity = 1 / (1 + dist)

                    results.append({
                        'text': self.documents[idx],
                        'similarity': float(similarity),
                        'rank': i + 1,
                        'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                    })

            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def get_context(self, query: str, top_k: int = 3) -> Tuple[List[str], float]:
        """
        Get context documents for RAG.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (context_texts, best_similarity_score)
        """
        results = self.search(query, top_k)

        if not results:
            return [], 0.0

        context_texts = [r['text'] for r in results]
        best_similarity = results[0]['similarity'] if results else 0.0

        return context_texts, best_similarity

    def rebuild_index(self, documents: List[Dict]) -> int:
        """
        Rebuild FAISS index from scratch.

        Args:
            documents: List of document dicts with 'text' and optional 'metadata'

        Returns:
            Total number of chunks indexed
        """
        # Clear existing index
        self.index = None
        self.documents = []
        self.document_metadata = []

        # Add all documents
        total_chunks = 0
        for doc in documents:
            chunks_added = self.add_document(
                doc['text'],
                doc.get('metadata')
            )
            total_chunks += chunks_added

        print(f"✓ Index rebuilt: {len(documents)} documents, {total_chunks} chunks")
        return total_chunks

    def get_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats
        """
        return {
            'total_documents': len(set(m.get('doc_id') for m in self.document_metadata if 'doc_id' in m)),
            'total_chunks': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_model': self.embedding_model_name,
            'index_path': self.index_path
        }

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple keyword-based search fallback when embeddings unavailable.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with text and similarity scores
        """
        if not self.documents:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each document by keyword overlap
        scored_docs = []
        for idx, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())

            # Calculate Jaccard similarity
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            similarity = intersection / union if union > 0 else 0

            # Boost exact phrase matches
            if query_lower in doc_lower:
                similarity += 0.3

            scored_docs.append({
                'idx': idx,
                'score': similarity
            })

        # Sort by score and take top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        top_docs = scored_docs[:top_k]

        # Format results
        results = []
        for rank, item in enumerate(top_docs):
            idx = item['idx']
            results.append({
                'text': self.documents[idx],
                'similarity': item['score'],
                'rank': rank + 1,
                'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
            })

        return results
