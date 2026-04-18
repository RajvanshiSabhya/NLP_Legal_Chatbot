# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories that need write access
RUN mkdir -p /app/data/raw /app/embeddings

# Create a non-root user (id 1000 is required by Hugging Face)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy the rest of the application code
COPY --chown=user . .

# Hugging Face Spaces expect 7860, Railway expects dynamic port.
# We don't hardcode EXPOSE to remain flexible for multiple platforms.

# Pre-download models to cache them in the Docker image
# This ensures faster startup and offline capability on Railway/HuggingFace
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering; \
    AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    AutoTokenizer.from_pretrained('deepset/roberta-base-squad2'); \
    AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')"

# Start the application using python to leverage the programmatic port handling in main.py
CMD ["python", "main.py"]
