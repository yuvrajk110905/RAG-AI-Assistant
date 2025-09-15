"""
Simple local vector store using numpy and basic similarity search.
No external vector database dependencies required.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Simple vector store using local files and numpy for similarity search."""
    
    def __init__(self, storage_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Files for storing vectors and metadata
        self.vectors_file = self.storage_path / "vectors.npy"
        self.metadata_file = self.storage_path / "metadata.json"
        self.model_file = self.storage_path / "model_info.json"
        
        # Initialize sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Load existing data
        self.vectors = self._load_vectors()
        self.metadata = self._load_metadata()
        
        # Save model info
        self._save_model_info(model_name)
        
    def _load_vectors(self) -> np.ndarray:
        """Load vectors from file."""
        if self.vectors_file.exists():
            try:
                return np.load(self.vectors_file)
            except Exception as e:
                logger.warning(f"Could not load vectors: {e}")
        
        # Return empty array with correct shape
        return np.empty((0, self.embedding_dim), dtype=np.float32)
    
    def _save_vectors(self):
        """Save vectors to file."""
        try:
            np.save(self.vectors_file, self.vectors)
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return []
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _save_model_info(self, model_name: str):
        """Save model information."""
        model_info = {
            "model_name": model_name,
            "embedding_dim": self.embedding_dim,
            "created_at": str(np.datetime64('now'))
        }
        try:
            with open(self.model_file, 'w') as f:
                json.dump(model_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model info: {e}")
    
    def add_texts(self, texts: List[str], metadata_list: List[Dict] = None) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Add to existing vectors
        if self.vectors.size == 0:
            self.vectors = embeddings.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, embeddings.astype(np.float32)])
        
        # Add metadata
        ids = []
        for i, text in enumerate(texts):
            text_id = f"text_{len(self.metadata)}"
            metadata = {
                "id": text_id,
                "text": text,
                "index": len(self.metadata),
                **(metadata_list[i] if metadata_list and i < len(metadata_list) else {})
            }
            self.metadata.append(metadata)
            ids.append(text_id)
        
        # Save to files
        self._save_vectors()
        self._save_metadata()
        
        logger.info(f"Added {len(texts)} texts to vector store")
        return ids
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """Search for similar texts."""
        if self.vectors.size == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.vectors)
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= score_threshold:
                results.append((self.metadata[idx], float(similarity_score)))
        
        return results
    
    def _cosine_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all vectors."""
        # Normalize vectors
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        similarities = np.dot(vectors_norm, query_norm)
        return similarities
    
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        """Get metadata by IDs."""
        results = []
        for metadata in self.metadata:
            if metadata.get("id") in ids:
                results.append(metadata)
        return results
    
    def delete_by_ids(self, ids: List[str]):
        """Delete vectors and metadata by IDs."""
        indices_to_remove = []
        for i, metadata in enumerate(self.metadata):
            if metadata.get("id") in ids:
                indices_to_remove.append(i)
        
        if indices_to_remove:
            # Remove from vectors (in reverse order to maintain indices)
            for idx in sorted(indices_to_remove, reverse=True):
                self.vectors = np.delete(self.vectors, idx, axis=0)
                del self.metadata[idx]
            
            # Update indices in metadata
            for i, metadata in enumerate(self.metadata):
                metadata["index"] = i
            
            self._save_vectors()
            self._save_metadata()
            
            logger.info(f"Deleted {len(indices_to_remove)} vectors")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            "total_vectors": len(self.vectors) if self.vectors.size > 0 else 0,
            "embedding_dimension": self.embedding_dim,
            "storage_path": str(self.storage_path),
            "model_info": self._get_model_info()
        }
    
    def _get_model_info(self) -> Dict:
        """Get model information."""
        if self.model_file.exists():
            try:
                with open(self.model_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"model_name": "unknown"}
    
    def clear(self):
        """Clear all vectors and metadata."""
        self.vectors = np.empty((0, self.embedding_dim), dtype=np.float32)
        self.metadata = []
        self._save_vectors()
        self._save_metadata()
        logger.info("Cleared all vectors and metadata")
