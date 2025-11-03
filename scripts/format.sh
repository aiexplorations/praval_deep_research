#!/bin/bash
# Code formatting script for Praval Deep Research
# Runs black and ruff --fix to auto-format code
set -e

echo "🎨 Formatting code..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if tools are installed
if ! command -v black &> /dev/null; then
    echo -e "${YELLOW}Warning: black is not installed${NC}"
    echo "Install with: pip install -r requirements-dev.txt"
    exit 1
fi

if ! command -v ruff &> /dev/null; then
    echo -e "${YELLOW}Warning: ruff is not installed${NC}"
    echo "Install with: pip install -r requirements-dev.txt"
    exit 1
fi

# Run black formatting
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Black code formatter"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
black src/ tests/ --line-length=88 --target-version=py39
echo -e "${GREEN}✓ Black formatting complete${NC}"
echo ""

# Run ruff auto-fixes
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Ruff auto-fix"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ruff check src/ tests/ --fix
echo -e "${GREEN}✓ Ruff auto-fix complete${NC}"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Code formatting complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Run linting to verify: ./scripts/lint.sh"
echo "  3. Run tests: ./scripts/test.sh"
echo "  4. Commit changes: git add -A && git commit -m 'Apply code formatting'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
