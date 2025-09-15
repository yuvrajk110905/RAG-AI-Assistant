#!/usr/bin/env python3
"""
Simple startup script for the AI Class Notes Assistant.
Run this to start the simplified version without complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    try:
        import openai
    except ImportError:
        missing.append("openai")
    
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n💡 Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_config():
    """Check configuration."""
    try:
        from config import get_simple_settings    settings = get_simple_settings()
    
    print("📋 Configuration:")
    print(f"   Data directory: {settings.DATA_DIR}")
    print(f"   Host: {settings.HOST}:{settings.PORT}")
    print(f"   OpenAI API Key: {'✅ Configured' if settings.OPENAI_API_KEY else '❌ Not configured'}")
    
    if not settings.OPENAI_API_KEY:
        print("\n⚠️  Warning: OPENAI_API_KEY not set")
        print("   Set it in .env file or environment variable")
        print("   Some features will not work without it")
    
    return True

def main():
    """Main startup function."""
    print("🚀 Starting AI Class Notes Assistant (Simple Version)")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    check_config()
    
    # Create data directories
    from config import get_simple_settings
    settings = get_simple_settings()
    
    print(f"\n📁 Creating data directories...")
    try:
        settings.DATA_DIR.mkdir(exist_ok=True)
        settings.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        settings.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        settings.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        print("   ✅ Directories created")
    except Exception as e:
        print(f"   ❌ Error creating directories: {e}")
        sys.exit(1)
    
    # Start the application
    print(f"\n🌐 Starting server...")
    print(f"   URL: http://{settings.HOST}:{settings.PORT}")
    print(f"   API Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   Health Check: http://{settings.HOST}:{settings.PORT}/health")
    print("\n📝 To stop the server, press Ctrl+C")
    print("=" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "main_simple:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
