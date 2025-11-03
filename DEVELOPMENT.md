# Development Guide

This guide covers the development workflow, code quality standards, and local setup for contributing to Praval Deep Research.

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/aiexplorations/praval_deep_research.git
cd praval_deep_research

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit:

```bash
pre-commit install
```

This will run ruff, black, mypy, and bandit checks automatically on `git commit`.

To run hooks manually on all files:

```bash
pre-commit run --all-files
```

## Code Quality Standards

This project enforces strict code quality standards aligned with PEP8 and production-grade Python development.

### Standards

- **Python Version**: 3.9+ (target: 3.9)
- **Line Length**: 88 characters (Black default)
- **Type Hints**: Required for all functions, parameters, and return values
- **Docstrings**: Required for all public modules, classes, and functions
- **Test Coverage**: 90%+ for agents, 85%+ for infrastructure

### Tools

- **Ruff**: Fast Python linter (replaces flake8, pylint, isort)
- **Black**: Uncompromising code formatter
- **MyPy**: Static type checker
- **Bandit**: Security vulnerability scanner
- **Pytest**: Testing framework with coverage

## Development Workflow

### Local Development Scripts

We provide convenient scripts for common development tasks:

#### Format Code

Auto-format code with black and auto-fix ruff issues:

```bash
./scripts/format.sh
```

This will:
- Run black on all Python files
- Run ruff --fix to auto-correct violations
- Display summary of changes

#### Lint Code

Check code quality without making changes:

```bash
./scripts/lint.sh
```

This runs:
- Ruff linting (PEP8 compliance)
- MyPy type checking
- Bandit security scanning

If any checks fail, the script exits with error code 1.

#### Run Tests

Run the test suite:

```bash
# All tests with coverage
./scripts/test.sh

# Unit tests only
./scripts/test.sh --unit

# Integration tests only
./scripts/test.sh --integration

# Run linting before tests
./scripts/test.sh --with-lint

# Verbose output
./scripts/test.sh --verbose

# Stop on first failure
./scripts/test.sh --fail-fast

# Combined options
./scripts/test.sh --unit --with-lint --verbose
```

### Typical Development Cycle

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code** following the agent design patterns in `CLAUDE.md`

3. **Format and lint** as you go:
   ```bash
   ./scripts/format.sh
   ./scripts/lint.sh
   ```

4. **Write tests** for your changes (TDD approach encouraged)

5. **Run tests locally**:
   ```bash
   ./scripts/test.sh --with-lint
   ```

6. **Commit changes** (pre-commit hooks will run automatically):
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

7. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **CI checks will run** on your PR automatically

## CI/CD Pipeline

### Automated Checks

When you create a PR to `main`, the following automated checks run:

#### Code Quality Job

Runs on Python 3.9, 3.10, 3.11:
- ✅ Ruff linting
- ✅ Black formatting check
- ✅ MyPy type checking
- ✅ Bandit security scan

#### Test Suite Job

Runs on Python 3.9, 3.10, 3.11 with services (RabbitMQ, Qdrant):
- ✅ Unit tests with coverage
- ✅ Integration tests
- ✅ Coverage report upload to Codecov

All checks must pass for PR approval.

### Interpreting CI Failures

#### Ruff Linting Failures

```
src/agents/research/paper_discovery.py:42:1: F401 [*] `sys` imported but unused
```

**Fix**: Remove unused imports or run `./scripts/format.sh`

#### Black Formatting Failures

```
would reformat src/agents/research/paper_discovery.py
```

**Fix**: Run `./scripts/format.sh` locally and commit changes

#### MyPy Type Errors

```
src/agents/research/paper_discovery.py:42: error: Function is missing a return type annotation
```

**Fix**: Add type hints to functions:
```python
def my_function(param: str) -> Dict[str, Any]:
    ...
```

#### Test Failures

```
FAILED tests/unit/test_paper_discovery.py::test_agent_creation
```

**Fix**: Review test output, fix code or update tests, run `./scripts/test.sh`

## Code Quality Best Practices

### Type Hints (Required)

```python
from typing import Dict, List, Optional, Any
from praval.types import Spore

@agent("my_agent", responds_to=["message"], memory=True)
def my_agent(spore: Spore) -> None:
    data: str = spore.knowledge.get("key", "default")
    results: List[Dict[str, Any]] = process_data(data)
    broadcast({"type": "result", "knowledge": {"results": results}})
```

### Docstrings (Required)

```python
def process_papers(papers: List[Dict[str, Any]]) -> List[str]:
    """
    Process research papers and extract titles.

    Args:
        papers: List of paper dictionaries with metadata

    Returns:
        List of paper titles

    Raises:
        ValueError: If papers list is empty
    """
    if not papers:
        raise ValueError("Papers list cannot be empty")

    return [paper["title"] for paper in papers]
```

### Agent Design (Praval Framework)

```python
@agent("research_specialist", responds_to=["search_request"], memory=True)
def research_specialist(spore: Spore) -> None:
    """I am a research specialist who discovers papers using learned patterns."""

    # Extract knowledge
    query = spore.knowledge.get("query")

    # Use memory for context
    past_searches = research_specialist.recall(query, limit=5)

    # Apply intelligence via LLM
    enhanced_query = chat(f"Enhance query: {query}")

    # Remember for learning
    research_specialist.remember(f"Processed {query}")

    # Broadcast results
    broadcast({"type": "papers_found", "knowledge": {"results": [...]}})
```

## Troubleshooting

### Pre-commit hooks failing

```bash
# Update hooks to latest versions
pre-commit autoupdate

# Clear cache and reinstall
pre-commit clean
pre-commit install --install-hooks
```

### MyPy cache issues

```bash
# Clear mypy cache
rm -rf .mypy_cache/
mypy src/ --config-file=pyproject.toml
```

### Ruff cache issues

```bash
# Clear ruff cache
rm -rf .ruff_cache/
ruff check src/ tests/
```

### Virtual environment issues

```bash
# Delete and recreate venv
rm -rf .venv/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Additional Resources

- **Project Guidelines**: See `CLAUDE.md` for comprehensive project standards
- **Architecture**: See `DESIGN.md` for system design details
- **Praval Framework**: https://github.com/hsaeed3/praval (agent framework documentation)
- **Ruff Docs**: https://docs.astral.sh/ruff/
- **Black Docs**: https://black.readthedocs.io/
- **MyPy Docs**: https://mypy.readthedocs.io/

## Getting Help

- Check existing issues: https://github.com/aiexplorations/praval_deep_research/issues
- Review CLAUDE.md for project-specific guidelines
- Ask questions in PR comments
- Consult the Praval framework documentation

## Contributing

1. Follow the code quality standards outlined in this guide
2. Ensure all pre-commit hooks pass
3. Write tests for new features
4. Update documentation as needed
5. Create descriptive commit messages
6. Ensure CI checks pass before requesting review

Thank you for contributing to Praval Deep Research!
