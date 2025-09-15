#!/bin/bash

# Quick start script for AI Class Notes Assistant (Simplified)
# Usage: ./quick_start.sh [your_openai_api_key]

echo "🎓 AI Class Notes Assistant - Quick Start"
echo "========================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install requirements
echo "📦 Installing requirements..."
cd backend
if ! python3 -m pip install -r requirements.txt; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Setup environment
if [ ! -f .env ]; then
    if [ -f .env.simple ]; then
        cp .env.simple .env
        echo "✅ Created .env from template"
    else
        echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
        echo "HOST=localhost" >> .env
        echo "PORT=8000" >> .env
        echo "✅ Created basic .env file"
    fi
fi
        echo "✅ Created basic .env file"
    fi
fi

# Set API key if provided
if [ ! -z "$1" ]; then
    sed -i "s/your_openai_api_key_here/$1/" .env
    echo "✅ OpenAI API key configured"
else
    echo "⚠️  Please edit .env and add your OpenAI API key"
fi

# Create data directory
mkdir -p data
echo "✅ Data directory created"

echo ""
echo "🚀 Starting AI Class Notes Assistant..."
echo "   Open http://localhost:8000 in your browser"
echo "   Use Ctrl+C to stop"
echo "----------------------------------------"

# Start the application
python3 main.py
