#!/bin/bash
# Linting script for Praval Deep Research
# Runs ruff, mypy, and bandit to check code quality
set -e

echo "🔍 Running code quality checks..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS=0

# Function to run a check and track status
run_check() {
    local name=$1
    local command=$2

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$command"; then
        echo -e "${GREEN}✓ $name passed${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
        OVERALL_STATUS=1
    fi
    echo ""
}

# Check if tools are installed
if ! command -v ruff &> /dev/null; then
    echo -e "${RED}Error: ruff is not installed${NC}"
    echo "Install with: pip install -r requirements-dev.txt"
    exit 1
fi

if ! command -v mypy &> /dev/null; then
    echo -e "${RED}Error: mypy is not installed${NC}"
    echo "Install with: pip install -r requirements-dev.txt"
    exit 1
fi

if ! command -v bandit &> /dev/null; then
    echo -e "${RED}Error: bandit is not installed${NC}"
    echo "Install with: pip install -r requirements-dev.txt"
    exit 1
fi

# Run ruff linting
run_check "Ruff linting (PEP8 compliance)" "ruff check src/ tests/"

# Run mypy type checking (only on src/, tests can be more flexible)
run_check "MyPy type checking" "mypy src/ --config-file=pyproject.toml"

# Run bandit security scanning
run_check "Bandit security scan" "bandit -c pyproject.toml -r src/ -q"

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ All code quality checks passed!${NC}"
else
    echo -e "${RED}✗ Some code quality checks failed${NC}"
    echo ""
    echo "To auto-fix some issues, run: ./scripts/format.sh"
    echo "For detailed output, run individual commands:"
    echo "  - ruff check src/ tests/ --show-fixes"
    echo "  - mypy src/ --config-file=pyproject.toml"
    echo "  - bandit -c pyproject.toml -r src/"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $OVERALL_STATUS
