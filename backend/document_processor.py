"""
Simple document processor for basic file types.
Handles PDF, TXT, and DOCX files without complex dependencies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

# Document processing imports
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleDocumentProcessor:
    """Simple document processor for basic file types."""
    
    def __init__(self):
        self.supported_formats = {'.txt': self._process_txt}
        
        if PDF_AVAILABLE:
            self.supported_formats['.pdf'] = self._process_pdf
        
        if DOCX_AVAILABLE:
            self.supported_formats['.docx'] = self._process_docx
            
        logger.info(f"Document processor initialized. Supported formats: {list(self.supported_formats.keys())}")
    
    def process_file(self, file_path: str, file_extension: Optional[str] = None) -> Dict:
        """Process a file and extract text content."""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file extension
        if not file_extension:
            file_extension = file_path_obj.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            # Process the file
            processor = self.supported_formats[file_extension]
            text = processor(file_path_obj)
            
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            return {
                "text": cleaned_text,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text),
                "file_type": file_extension,
                "word_count": len(cleaned_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _process_txt(self, file_path: Path) -> str:
        """Process a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _process_pdf(self, file_path: Path) -> str:
        """Process a PDF file using pdfplumber."""
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber not available. Install with: pip install pdfplumber")
        
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
        
        return text
    
    def _process_docx(self, file_path: Path) -> str:
        """Process a DOCX file using python-docx."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        
        text = ""
        try:
            doc = Document(str(file_path))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks for processing."""
        if not text:
            return []
        
        # Split by sentences first (basic approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Handle very long sentences that exceed chunk_size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # Split long chunks by character
                words = chunk.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            final_chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    final_chunks.append(temp_chunk)
        
        return final_chunks
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract basic keywords from text."""
        if not text:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'around', 'under', 'over'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:max_keywords]]
