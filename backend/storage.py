"""
Simple JSON-based storage system
Replaces MongoDB with local JSON files for document metadata and content.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class SimpleStorage:
    """Simple JSON-based storage for documents and metadata."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.documents_file = self.storage_path / "documents.json"
        self.chunks_file = self.storage_path / "chunks.json"
        self.chat_history_file = self.storage_path / "chat_history.json"
        self.generated_content_file = self.storage_path / "generated_content.json"
        
        # Initialize storage files if they don't exist
        self._init_storage_files()
        
    def _init_storage_files(self):
        """Initialize JSON storage files."""
        for file_path in [self.documents_file, self.chunks_file, self.chat_history_file, self.generated_content_file]:
            if not file_path.exists():
                self._save_json(file_path, [])
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_json(self, file_path: Path, data: List[Dict]):
        """Save data to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
    
    def add_document(self, file_path: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to storage."""
        doc_id = str(uuid.uuid4())
        
        document = {
            "id": doc_id,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        documents = self._load_json(self.documents_file)
        documents.append(document)
        self._save_json(self.documents_file, documents)
        
        logger.info(f"Added document: {document['file_name']}")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a document by ID."""
        documents = self._load_json(self.documents_file)
        for doc in documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents."""
        return self._load_json(self.documents_file)
    
    def search_documents(self, query: str) -> List[Dict]:
        """Simple text-based search in documents."""
        documents = self._load_json(self.documents_file)
        results = []
        
        query_lower = query.lower()
        for doc in documents:
            # Search in content and filename
            if (query_lower in doc.get("content", "").lower() or 
                query_lower in doc.get("file_name", "").lower()):
                results.append(doc)
        
        return results
    
    def add_chunks(self, doc_id: str, chunks: List[Dict]) -> List[str]:
        """Add text chunks for a document."""
        chunk_ids = []
        all_chunks = self._load_json(self.chunks_file)
        
        for i, chunk_data in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk = {
                "id": chunk_id,
                "document_id": doc_id,
                "content": chunk_data.get("content", ""),
                "chunk_index": i,
                "metadata": chunk_data.get("metadata", {}),
                "created_at": datetime.now().isoformat()
            }
            all_chunks.append(chunk)
            chunk_ids.append(chunk_id)
        
        self._save_json(self.chunks_file, all_chunks)
        return chunk_ids
    
    def get_chunks_by_document(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document."""
        all_chunks = self._load_json(self.chunks_file)
        return [chunk for chunk in all_chunks if chunk["document_id"] == doc_id]
    
    def add_chat_message(self, user_message: str, ai_response: str, context: Optional[Dict] = None):
        """Add a chat interaction to history."""
        chat_history = self._load_json(self.chat_history_file)
        
        interaction = {
            "id": str(uuid.uuid4()),
            "user_message": user_message,
            "ai_response": ai_response,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        chat_history.append(interaction)
        
        # Keep only last 100 interactions to prevent file from growing too large
        if len(chat_history) > 100:
            chat_history = chat_history[-100:]
        
        self._save_json(self.chat_history_file, chat_history)
    
    def get_chat_history(self, limit: int = 20) -> List[Dict]:
        """Get recent chat history."""
        chat_history = self._load_json(self.chat_history_file)
        return chat_history[-limit:] if chat_history else []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        # Delete document
        documents = self._load_json(self.documents_file)
        documents = [doc for doc in documents if doc["id"] != doc_id]
        self._save_json(self.documents_file, documents)
        
        # Delete associated chunks
        chunks = self._load_json(self.chunks_file)
        chunks = [chunk for chunk in chunks if chunk["document_id"] != doc_id]
        self._save_json(self.chunks_file, chunks)
        
        logger.info(f"Deleted document: {doc_id}")
        return True
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        documents = self._load_json(self.documents_file)
        chunks = self._load_json(self.chunks_file)
        chat_history = self._load_json(self.chat_history_file)
        
        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "chat_interactions": len(chat_history),
            "storage_path": str(self.storage_path),
            "last_updated": datetime.now().isoformat()
        }
    
    def add_generated_content(self, content_data: Dict) -> str:
        """Add generated content (study plans, summaries, etc.)."""
        content_id = str(uuid.uuid4())
        content_item = {
            "id": content_id,
            "created_at": datetime.now().isoformat(),
            **content_data
        }
        
        generated_content = self._load_json(self.generated_content_file)
        generated_content.append(content_item)
        self._save_json(self.generated_content_file, generated_content)
        
        logger.info(f"Added generated content: {content_data.get('type', 'unknown')}")
        return content_id
    
    def get_generated_content(self, content_type: Optional[str] = None) -> List[Dict]:
        """Get generated content, optionally filtered by type."""
        content = self._load_json(self.generated_content_file)
        
        if content_type:
            content = [item for item in content if item.get("type") == content_type]
        
        # Sort by creation date, newest first
        content.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return content
    
    def delete_generated_content(self, content_id: str) -> bool:
        """Delete generated content by ID."""
        content = self._load_json(self.generated_content_file)
        original_count = len(content)
        content = [item for item in content if item["id"] != content_id]
        
        if len(content) < original_count:
            self._save_json(self.generated_content_file, content)
            logger.info(f"Deleted generated content: {content_id}")
            return True
        return False
