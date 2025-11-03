#!/bin/bash
# Test runner script for Agentic Deep Research
# This is what I want to do: Run comprehensive unit tests before deployment
# So I think this is how it goes: Execute pytest with proper configuration
# And that goes there: In the scripts directory for easy access
# So I changed this: Created a new test runner that validates code before build

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}Agentic Deep Research - Test Suite${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install/update dependencies
echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -q -r requirements.txt
pip install -q -r requirements-dev.txt
echo -e "${GREEN}Dependencies installed${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="all"
COVERAGE=true
VERBOSE=false
FAIL_FAST=false
WITH_LINT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --with-lint)
            WITH_LINT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--unit|--integration] [--no-coverage] [--verbose] [--fail-fast] [--with-lint]"
            exit 1
            ;;
    esac
done

# Run linting if requested
if [ "$WITH_LINT" = true ]; then
    echo -e "${BLUE}Running code quality checks before tests...${NC}"
    echo ""

    if [ -f "./scripts/lint.sh" ]; then
        ./scripts/lint.sh
        LINT_EXIT_CODE=$?

        if [ $LINT_EXIT_CODE -ne 0 ]; then
            echo ""
            echo -e "${RED}Linting failed. Fix code quality issues before running tests.${NC}"
            exit 1
        fi

        echo ""
        echo -e "${GREEN}✓ All linting checks passed. Proceeding with tests...${NC}"
        echo ""
    else
        echo -e "${YELLOW}Warning: lint.sh not found, skipping linting${NC}"
        echo ""
    fi
fi

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$FAIL_FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -x"
fi

# Add test markers based on type
if [ "$TEST_TYPE" = "unit" ]; then
    PYTEST_CMD="$PYTEST_CMD -m unit"
    echo -e "${BLUE}Running unit tests only...${NC}"
elif [ "$TEST_TYPE" = "integration" ]; then
    PYTEST_CMD="$PYTEST_CMD -m integration"
    echo -e "${BLUE}Running integration tests only...${NC}"
else
    echo -e "${BLUE}Running all tests...${NC}"
fi

# Add coverage if enabled
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

echo ""
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

# Run tests
$PYTEST_CMD

TEST_EXIT_CODE=$?

echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}===========================================${NC}"

    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${YELLOW}Coverage report generated: htmlcov/index.html${NC}"
    fi

    echo ""
    echo -e "${GREEN}Code is ready for deployment!${NC}"
else
    echo -e "${RED}===========================================${NC}"
    echo -e "${RED}✗ Tests failed!${NC}"
    echo -e "${RED}===========================================${NC}"
    echo ""
    echo -e "${RED}Fix the failing tests before deploying.${NC}"
    exit 1
fi

exit 0
