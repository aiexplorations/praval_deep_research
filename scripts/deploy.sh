#!/bin/bash
# Deploy script for Agentic Deep Research

set -e  # Exit on error

# Colors for output
GREEN='\033[0.32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Agentic Deep Research - Deployment${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo -e "${YELLOW}Copying .env.example to .env...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env and add your OPENAI_API_KEY${NC}"
    echo -e "${RED}Then run this script again.${NC}"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if grep -q "your_openai_api_key_here" .env; then
    echo -e "${RED}Error: Please set your OPENAI_API_KEY in .env file${NC}"
    exit 1
fi

# Parse command line arguments
WITH_FRONTEND=false
WITH_MONITORING=false
BUILD_IMAGES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-frontend)
            WITH_FRONTEND=true
            shift
            ;;
        --with-monitoring)
            WITH_MONITORING=true
            shift
            ;;
        --no-build)
            BUILD_IMAGES=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--with-frontend] [--with-monitoring] [--no-build]"
            exit 1
            ;;
    esac
done

# Build Docker images
if [ "$BUILD_IMAGES" = true ]; then
    echo -e "${GREEN}Building Docker images...${NC}"
    docker-compose build
    echo ""
fi

# Start infrastructure services
echo -e "${GREEN}Starting infrastructure services...${NC}"
docker-compose up -d rabbitmq qdrant minio redis
echo ""

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check service health
echo -e "${GREEN}Checking service health...${NC}"
docker-compose ps
echo ""

# Start application services
echo -e "${GREEN}Starting application services...${NC}"
docker-compose up -d research_api research_agents
echo ""

# Start optional services
if [ "$WITH_FRONTEND" = true ]; then
    echo -e "${GREEN}Starting frontend...${NC}"
    docker-compose --profile with-frontend up -d research_frontend
    echo ""
fi

if [ "$WITH_MONITORING" = true ]; then
    echo -e "${GREEN}Starting monitoring...${NC}"
    docker-compose --profile monitoring up -d prometheus
    echo ""
fi

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}API is ready!${NC}"
        break
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}API failed to start within expected time${NC}"
    echo -e "${YELLOW}Check logs with: docker-compose logs research_api${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "${GREEN}Services:${NC}"
echo "  - API:          http://localhost:8000"
echo "  - API Docs:     http://localhost:8000/docs"
echo "  - RabbitMQ UI:  http://localhost:15672 (user: research_user, pass: research_pass)"
echo "  - MinIO Console: http://localhost:9001 (user: minioadmin, pass: minioadmin)"

if [ "$WITH_FRONTEND" = true ]; then
    echo "  - Frontend:     http://localhost:3000"
fi

if [ "$WITH_MONITORING" = true ]; then
    echo "  - Prometheus:   http://localhost:9090"
fi

echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo "  docker-compose logs -f research_api"
echo "  docker-compose logs -f research_agents"
echo ""
echo -e "${YELLOW}To stop services:${NC}"
echo "  ./scripts/stop.sh"
echo ""
