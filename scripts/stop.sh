#!/bin/bash
# Stop script for Agentic Deep Research

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Stopping Agentic Deep Research${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""

# Stop all services
echo -e "${YELLOW}Stopping all services...${NC}"
docker-compose --profile with-frontend --profile monitoring --profile dev-tools --profile with-postgres down

echo ""
echo -e "${GREEN}All services stopped successfully!${NC}"
echo ""
echo -e "${YELLOW}Note: Data volumes are preserved.${NC}"
echo -e "${YELLOW}To remove volumes, use: ./scripts/clean.sh${NC}"
echo ""
