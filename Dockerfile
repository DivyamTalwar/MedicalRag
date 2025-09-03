FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_chatbot/ ./rag_chatbot/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/data/embeddings /app/data/documents

# Set environment variables
ENV PYTHONPATH=/app
ENV APP_ENV=production

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "rag_chatbot.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]