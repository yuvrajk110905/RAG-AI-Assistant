"""
Simplified AI Class Notes Assistant
Lightweight version with minimal dependencies - no MongoDB, Redis, or complex scheduling.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

# Import our simplified modules
from config import get_simple_settings
from storage import SimpleStorage
from vector_store import SimpleVectorStore
from llm import SimpleLLMInterface
from document_processor import SimpleDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_simple_settings()

# Initialize components
storage = SimpleStorage(settings.DATA_DIR / "storage")
vector_store = SimpleVectorStore(settings.VECTOR_STORE_PATH)
llm = SimpleLLMInterface(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
doc_processor = SimpleDocumentProcessor()

# FastAPI app
app = FastAPI(
    title="AI Class Notes Assistant (Simple)",
    description="Lightweight AI assistant for academic support",
    version="1.0.0-simple"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    use_context: bool = True

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class StudyPlanRequest(BaseModel):
    subject: str
    topics: list[str] = []
    deadline: str

class AssignmentHelpRequest(BaseModel):
    assignment_description: str
    requirements: str = ""

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 500

class ConceptExplanationRequest(BaseModel):
    concept: str
    use_context: bool = True

# Routes
@app.get("/")
async def root():
    """Serve the main frontend page."""
    return FileResponse("static/index.html")

@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "message": "AI Class Notes Assistant (Simple Version)",
        "version": "1.0.0-simple",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = storage.get_storage_stats()
    vector_stats = vector_store.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "storage": "healthy",
            "vector_store": "healthy",
            "llm": "healthy" if llm else "not configured",
            "documents": stats["total_documents"],
            "vectors": vector_stats["total_vectors"]
        }
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form(default="")
):
    """Upload and process a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    file_extension = Path(file.filename).suffix.lower().lstrip('.')  # Remove the dot
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type .{file_extension} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Process document
        processed_content = doc_processor.process_file(temp_path, f'.{file_extension}')
        
        # Save to storage
        metadata = {
            "subject": subject,
            "file_size": len(content),
            "file_type": f'.{file_extension}'
        }
        doc_id = storage.add_document(file.filename, processed_content["text"], metadata)
        
        # Create chunks and add to vector store
        chunks = doc_processor.create_chunks(processed_content["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        chunk_metadata = [{"document_id": doc_id, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
        
        # Add chunks to storage
        storage.add_chunks(doc_id, [{"content": chunk} for chunk in chunks])
        
        # Add to vector store
        vector_store.add_texts(chunks, chunk_metadata)
        
        # Clean up
        Path(temp_path).unlink()
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "text_length": len(processed_content["text"])
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Chat with the AI assistant."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not configured. Please set OPENAI_API_KEY")
    
    try:
        context_text = ""
        
        # Get relevant context if requested
        if request.use_context:
            logger.info(f"Searching for context with query: {request.message}")
            search_results = vector_store.similarity_search(request.message, k=3)
            logger.info(f"Search results count: {len(search_results)}")
            if search_results:
                logger.info(f"First result structure: {search_results[0]}")
                try:
                    context_chunks = [result["text"] for result, score in search_results]
                    context_text = "\n\n".join(context_chunks)
                    logger.info(f"Context text length: {len(context_text)}")
                except Exception as e:
                    logger.error(f"Error extracting context chunks: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
        
        # Generate response
        logger.info(f"Generating response with LLM")
        try:
            response = llm.generate_response(request.message, context_text)
            logger.info(f"LLM response generated successfully")
        except Exception as e:
            logger.error(f"Error in LLM generate_response: {e}")
            raise
        
        # Save to chat history
        storage.add_chat_message(
            request.message, 
            response, 
            {"context_used": bool(context_text), "context_length": len(context_text)}
        )
        
        return {
            "response": response,
            "context_used": bool(context_text),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search through uploaded documents."""
    try:
        # Text-based search in storage
        text_results = storage.search_documents(request.query)
        
        # Vector-based search
        vector_results = vector_store.similarity_search(request.query, k=request.limit)
        
        # Combine and format results
        combined_results = []
        
        # Add text search results
        for doc in text_results[:request.limit//2]:
            combined_results.append({
                "type": "document",
                "id": doc["id"],
                "title": doc["file_name"],
                "content": doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"],
                "metadata": doc["metadata"],
                "score": 1.0  # Default score for text search
            })
        
        # Add vector search results
        for result, score in vector_results[:request.limit//2]:
            combined_results.append({
                "type": "chunk",
                "content": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "score": score,
                "metadata": result
            })
        
        return {
            "results": combined_results,
            "total_found": len(combined_results),
            "query": request.query
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    try:
        documents = storage.get_all_documents()
        return {
            "documents": [
                {
                    "id": doc["id"],
                    "filename": doc["file_name"],
                    "created_at": doc["created_at"],
                    "metadata": doc["metadata"]
                }
                for doc in documents
            ],
            "total": len(documents)
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    try:
        # Get document chunks to remove from vector store
        chunks = storage.get_chunks_by_document(doc_id)
        chunk_ids = [chunk["id"] for chunk in chunks]
        
        # Remove from vector store
        if chunk_ids:
            vector_store.delete_by_ids(chunk_ids)
        
        # Remove from storage
        success = storage.delete_document(doc_id)
        
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_chat_history(limit: int = 20):
    """Get chat history."""
    try:
        history = storage.get_chat_history(limit)
        return {"history": history, "total": len(history)}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        storage_stats = storage.get_storage_stats()
        vector_stats = vector_store.get_stats()
        
        return {
            "storage": storage_stats,
            "vector_store": vector_stats,
            "system": {
                "data_directory": str(settings.DATA_DIR),
                "max_file_size": settings.MAX_FILE_SIZE,
                "allowed_extensions": list(settings.ALLOWED_EXTENSIONS),
                "chunk_size": settings.CHUNK_SIZE
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Agent Endpoints
@app.post("/study-plan")
async def generate_study_plan(request: StudyPlanRequest):
    """Generate a study plan using AI."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not configured. Please set OPENAI_API_KEY")
    
    try:
        plan = llm.generate_study_plan(request.subject, request.topics, request.deadline)
        
        # Save the generated plan
        plan_data = {
            "type": "study_plan",
            "subject": request.subject,
            "topics": request.topics,
            "deadline": request.deadline,
            "content": plan,
            "created_at": datetime.now().isoformat()
        }
        storage.add_generated_content(plan_data)
        
        return {
            "plan": plan,
            "subject": request.subject,
            "deadline": request.deadline,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating study plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assignment-help")
async def get_assignment_help(request: AssignmentHelpRequest):
    """Get help with assignment planning."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not configured. Please set OPENAI_API_KEY")
    
    try:
        help_content = llm.help_with_assignment(request.assignment_description, request.requirements)
        
        # Save the generated help
        help_data = {
            "type": "assignment_help",
            "assignment_description": request.assignment_description,
            "requirements": request.requirements,
            "content": help_content,
            "created_at": datetime.now().isoformat()
        }
        storage.add_generated_content(help_data)
        
        return {
            "help": help_content,
            "assignment": request.assignment_description,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating assignment help: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """Summarize text using AI."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not configured. Please set OPENAI_API_KEY")
    
    try:
        summary = llm.summarize_text(request.text, request.max_length)
        
        # Save the summary
        summary_data = {
            "type": "summary",
            "original_length": len(request.text),
            "summary_length": len(summary),
            "content": summary,
            "created_at": datetime.now().isoformat()
        }
        storage.add_generated_content(summary_data)
        
        return {
            "summary": summary,
            "original_length": len(request.text),
            "summary_length": len(summary),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain-concept")
async def explain_concept(request: ConceptExplanationRequest):
    """Explain a concept using AI with optional context."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not configured. Please set OPENAI_API_KEY")
    
    try:
        context = ""
        if request.use_context:
            # Get relevant context from uploaded documents
            search_results = vector_store.similarity_search(request.concept, k=3)
            if search_results:
                context_chunks = [result["text"] for result, score in search_results]
                context = "\n\n".join(context_chunks)
        
        explanation = llm.explain_concept(request.concept, context)
        
        # Save the explanation
        explanation_data = {
            "type": "concept_explanation",
            "concept": request.concept,
            "used_context": bool(context),
            "content": explanation,
            "created_at": datetime.now().isoformat()
        }
        storage.add_generated_content(explanation_data)
        
        return {
            "explanation": explanation,
            "concept": request.concept,
            "context_used": bool(context),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error explaining concept: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generated-content")
async def get_generated_content():
    """Get all generated content (study plans, summaries, etc.)."""
    try:
        content = storage.get_generated_content()
        return {
            "content": content,
            "total": len(content),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting generated content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    logger.info("Starting AI Class Notes Assistant (Simple Version)")
    logger.info(f"Data directory: {settings.DATA_DIR}")
    logger.info(f"OpenAI API configured: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
