#!/bin/bash
# End-to-end test script for Agentic Deep Research
# This script tests the complete workflow: Search → Select → Index → Q&A

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="http://localhost:8000"

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}End-to-End Workflow Test${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}Test 1: API Health Check${NC}"
HEALTH=$(curl -s ${API_URL}/health/ | jq -r '.status')
if [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "degraded" ]; then
    echo -e "${GREEN}✓ API is ${HEALTH}${NC}"
else
    echo -e "${RED}✗ API health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: Search Papers
echo -e "${YELLOW}Test 2: Searching for papers on 'machine learning transformers'${NC}"
SEARCH_RESPONSE=$(curl -s -X POST ${API_URL}/research/search \
    -H "Content-Type: application/json" \
    -d '{
        "query": "machine learning transformers",
        "domain": "artificial intelligence",
        "max_results": 5,
        "quality_threshold": 0.3
    }')

PAPERS_COUNT=$(echo $SEARCH_RESPONSE | jq '.papers | length')
echo -e "${GREEN}✓ Found ${PAPERS_COUNT} papers${NC}"

if [ "$PAPERS_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No papers found${NC}"
    exit 1
fi

# Extract first 2 papers for indexing
PAPERS_TO_INDEX=$(echo $SEARCH_RESPONSE | jq '.papers[0:2]')
echo ""

# Test 3: Index Selected Papers
echo -e "${YELLOW}Test 3: Indexing 2 selected papers${NC}"
INDEX_RESPONSE=$(curl -s -X POST ${API_URL}/research/index \
    -H "Content-Type: application/json" \
    -d "{\"papers\": ${PAPERS_TO_INDEX}}")

echo "$INDEX_RESPONSE" | jq '.'
echo -e "${GREEN}✓ Papers submitted for indexing${NC}"
echo ""

# Wait for indexing to complete
echo -e "${YELLOW}Waiting 30 seconds for indexing to complete...${NC}"
sleep 30

# Test 4: Check Qdrant for vectors
echo -e "${YELLOW}Test 4: Checking Qdrant for stored vectors${NC}"
VECTOR_COUNT=$(curl -s http://localhost:6333/collections/research_vectors | jq '.result.points_count')
echo -e "${GREEN}✓ Qdrant has ${VECTOR_COUNT} vectors stored${NC}"

if [ "$VECTOR_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No vectors found in Qdrant - indexing may have failed${NC}"
    echo ""
    echo -e "${YELLOW}Checking document processor logs:${NC}"
    docker-compose logs research_agents --tail 50 | grep -A5 -B5 "document_processor\|papers_found\|Qdrant"
    exit 1
fi
echo ""

# Test 5: Q&A Test
echo -e "${YELLOW}Test 5: Asking question about indexed papers${NC}"
QA_RESPONSE=$(curl -s -X POST ${API_URL}/research/ask \
    -H "Content-Type: application/json" \
    -d '{
        "question": "What are transformers in machine learning?",
        "include_sources": true
    }')

ANSWER=$(echo $QA_RESPONSE | jq -r '.answer')
SOURCES_COUNT=$(echo $QA_RESPONSE | jq '.sources | length')

if [ -z "$ANSWER" ] || [ "$ANSWER" = "null" ]; then
    echo -e "${RED}✗ No answer received${NC}"
    echo "$QA_RESPONSE" | jq '.'
    exit 1
fi

echo -e "${GREEN}✓ Received answer with ${SOURCES_COUNT} sources${NC}"
echo ""
echo -e "${BLUE}Answer:${NC}"
echo "$ANSWER" | head -c 500
echo "..."
echo ""

# Test 6: Follow-up Questions
FOLLOWUP_COUNT=$(echo $QA_RESPONSE | jq '.followup_questions | length')
if [ "$FOLLOWUP_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Received ${FOLLOWUP_COUNT} follow-up questions${NC}"
else
    echo -e "${YELLOW}⚠ No follow-up questions generated${NC}"
fi
echo ""

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}✓ All tests passed!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  - Papers found: ${PAPERS_COUNT}"
echo "  - Papers indexed: 2"
echo "  - Vectors stored: ${VECTOR_COUNT}"
echo "  - Q&A sources: ${SOURCES_COUNT}"
echo "  - Follow-up questions: ${FOLLOWUP_COUNT}"
echo ""
echo -e "${GREEN}End-to-end workflow is functioning correctly!${NC}"
