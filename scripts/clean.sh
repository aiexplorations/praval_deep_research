#!/bin/bash
# Clean script for Agentic Deep Research
# WARNING: This removes all data volumes

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}===========================================${NC}"
echo -e "${RED}WARNING: This will remove ALL data!${NC}"
echo -e "${RED}===========================================${NC}"
echo ""
echo "This will:"
echo "  - Stop all containers"
echo "  - Remove all containers"
echo "  - Remove all volumes (PDFs, vectors, cache, etc.)"
echo "  - Remove all networks"
echo ""
echo -e "${YELLOW}Data that will be lost:${NC}"
echo "  - All uploaded PDFs in MinIO"
echo "  - All vector embeddings in Qdrant"
echo "  - All cached data in Redis"
echo "  - All RabbitMQ messages"
echo "  - All Prometheus metrics"
echo ""

# Prompt for confirmation
read -p "Are you sure you want to continue? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo -e "${GREEN}Operation cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Stopping and removing all services...${NC}"
docker-compose --profile with-frontend --profile monitoring --profile dev-tools --profile with-postgres down -v

echo ""
echo -e "${YELLOW}Removing unused Docker resources...${NC}"
docker system prune -f

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "All containers, volumes, and networks have been removed."
echo ""
echo -e "${YELLOW}To start fresh:${NC}"
echo "  ./scripts/deploy.sh"
echo ""
