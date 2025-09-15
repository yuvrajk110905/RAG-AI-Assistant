#!/bin/bash

# Quick Setup Script for AI Class Notes Assistant
# This script helps you get started quickly with the application

set -e

echo "🎓 AI Class Notes Assistant - Quick Setup"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies
echo "📚 Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Create .env file if it doesn't exist
if [ ! -f backend/.env ]; then
    echo "📝 Creating .env file..."
    cp backend/.env.example backend/.env
    echo "⚠️  Please edit backend/.env with your OpenAI API key and other configuration"
fi

# Check if MongoDB is running
echo "🔍 Checking MongoDB..."
if command -v docker &> /dev/null; then
    echo "🐳 Docker found. Starting MongoDB with Docker..."
    docker run -d -p 27017:27017 --name mongodb-class-notes mongo:latest 2>/dev/null || echo "MongoDB container already running or failed to start"
else
    echo "⚠️  Docker not found. Please install and start MongoDB manually:"
    echo "   - Install MongoDB: https://docs.mongodb.com/manual/installation/"
    echo "   - Or use Docker: docker run -d -p 27017:27017 --name mongodb mongo:latest"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p backend/uploads
mkdir -p backend/generated_documents
mkdir -p backend/logs
mkdir -p backend/data/vector_store

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit backend/.env with your OpenAI API key"
echo "2. Make sure MongoDB is running"
echo "3. Start the application:"
echo "   cd backend && python run_server.py"
echo ""
echo "🌐 Access the application:"
echo "   - Frontend: Open frontend/index.html in your browser"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Need help? Check the README.md file for detailed instructions."
