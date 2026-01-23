#!/bin/bash
# Migration Script: Move data from Docker volumes to local bind mounts
# This script copies existing data from Docker named volumes to local ./data/ folders
# Run this BEFORE switching to the new docker-compose.yml with bind mounts

set -e

echo "=============================================="
echo "Praval Deep Research - Data Migration Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Create data directories if they don't exist
echo -e "${YELLOW}Creating local data directories...${NC}"
mkdir -p "$DATA_DIR/minio"
mkdir -p "$DATA_DIR/qdrant"
mkdir -p "$DATA_DIR/vajra_indexes"
mkdir -p "$DATA_DIR/postgres"
mkdir -p "$DATA_DIR/neo4j/data"
mkdir -p "$DATA_DIR/neo4j/logs"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Function to migrate a Docker volume
migrate_volume() {
    local volume_name=$1
    local dest_path=$2
    local container_path=$3

    echo -e "${YELLOW}Migrating $volume_name...${NC}"

    # Check if volume exists
    if ! docker volume inspect "$volume_name" > /dev/null 2>&1; then
        echo -e "${RED}  Volume $volume_name does not exist. Skipping.${NC}"
        return 0
    fi

    # Check if destination already has data
    if [ "$(ls -A "$dest_path" 2>/dev/null)" ]; then
        echo -e "${YELLOW}  Warning: $dest_path already contains data.${NC}"
        read -p "  Overwrite existing data? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "  Skipping $volume_name"
            return 0
        fi
    fi

    # Copy data using a temporary container
    echo "  Copying data from Docker volume to local folder..."
    docker run --rm \
        -v "$volume_name:$container_path:ro" \
        -v "$dest_path:/dest" \
        alpine sh -c "cp -av $container_path/. /dest/" 2>&1 | head -20

    # Count files
    local file_count=$(find "$dest_path" -type f 2>/dev/null | wc -l)
    echo -e "${GREEN}  ✓ Migrated $file_count files${NC}"
}

# Migrate MinIO (research papers)
echo "=============================================="
echo "Step 1: Migrating MinIO (Research Papers)"
echo "=============================================="
migrate_volume "praval_deep_research_minio_data" "$DATA_DIR/minio" "/data"
echo ""

# After migration, papers will be at:
# ./data/minio/research-papers/<uuid>.pdf
# ./data/minio/.minio.sys/ (MinIO metadata)

# Migrate Qdrant (vector embeddings)
echo "=============================================="
echo "Step 2: Migrating Qdrant (Vector Embeddings)"
echo "=============================================="
migrate_volume "praval_deep_research_qdrant_data" "$DATA_DIR/qdrant" "/qdrant/storage"
echo ""

# Migrate Vajra indexes (BM25 search)
echo "=============================================="
echo "Step 3: Migrating Vajra Indexes (BM25 Search)"
echo "=============================================="
migrate_volume "praval_deep_research_vajra_data" "$DATA_DIR/vajra_indexes" "/app/data/vajra_indexes"
echo ""

# Migrate PostgreSQL (chat history, metadata)
echo "=============================================="
echo "Step 4: Migrating PostgreSQL (Metadata)"
echo "=============================================="
echo -e "${YELLOW}Note: PostgreSQL requires special handling for cross-platform compatibility${NC}"
echo "For PostgreSQL, we recommend using pg_dump instead of raw file copy."
echo ""
echo "To migrate PostgreSQL data:"
echo "  1. With containers running, execute:"
echo "     docker exec research_postgres pg_dump -U research_user praval_research > backup.sql"
echo "  2. After switching to bind mounts, restore with:"
echo "     docker exec -i research_postgres psql -U research_user praval_research < backup.sql"
echo ""

# Migrate Neo4j (knowledge graph)
echo "=============================================="
echo "Step 5: Migrating Neo4j (Knowledge Graph)"
echo "=============================================="
migrate_volume "praval_deep_research_neo4j_data" "$DATA_DIR/neo4j/data" "/data"
migrate_volume "praval_deep_research_neo4j_logs" "$DATA_DIR/neo4j/logs" "/logs"
echo ""

echo "=============================================="
echo "Migration Summary"
echo "=============================================="
echo ""
echo "Local data directory structure:"
find "$DATA_DIR" -maxdepth 2 -type d | head -20
echo ""
echo "File counts:"
echo "  MinIO (papers):    $(find "$DATA_DIR/minio" -type f -name "*.pdf" 2>/dev/null | wc -l) PDFs"
echo "  Qdrant (vectors):  $(find "$DATA_DIR/qdrant" -type f 2>/dev/null | wc -l) files"
echo "  Vajra (indexes):   $(find "$DATA_DIR/vajra_indexes" -type f 2>/dev/null | wc -l) files"
echo "  Neo4j (graph):     $(find "$DATA_DIR/neo4j" -type f 2>/dev/null | wc -l) files"
echo ""
echo -e "${GREEN}=============================================="
echo "Migration complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Stop the current containers: docker compose down"
echo "  2. The new docker-compose.yml uses bind mounts to ./data/"
echo "  3. Start with new config: docker compose up -d"
echo "  4. Your papers are now at: ./data/minio/research-papers/"
echo -e "${NC}"
