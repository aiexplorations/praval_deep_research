#!/bin/bash
set -e

echo "=== Praval Deep Research API Startup ==="
echo "Initializing Vajra BM25 indexes..."

# Run index initialization (creates indexes if they don't exist)
python /app/scripts/init_vajra_indexes.py

echo "Index initialization complete."
echo "Starting FastAPI server..."

# Start the API server
exec python -m uvicorn agentic_research.api.main:app --host 0.0.0.0 --port 8000
