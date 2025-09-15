FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_simple.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_simple.txt

# Copy application files
COPY config_simple.py .
COPY simple_storage.py .
COPY simple_vector_store.py .
COPY simple_llm.py .
COPY simple_document_processor.py .
COPY main_simple.py .
COPY run_simple.py .
COPY frontend_simple.html .

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "main.py"]
