# Praval Deep Research - Project Guidelines

## Project Overview

The Praval Deep Research system is an intelligent research assistant built on the Praval agentic framework. This project demonstrates excellence in agent-driven architecture, combining academic rigor with enterprise-grade engineering practices.

### Core Architecture Principles

1. **Praval-First Design**: All intelligent behavior is implemented through proper Praval agents
2. **Identity-Driven Agents**: Agents are defined by what they ARE, not what they DO
3. **Test-Driven Development**: Comprehensive testing drives all development decisions
4. **Production-Ready Quality**: Every component meets enterprise standards
5. **Academic Rigor**: Research functionality grounded in sound theoretical foundations
6. **No mock code or fake implementations**: All code is real and there is no simulation, mocking or other such stuff

## Praval Agent Development Standards

### Agent Design Philosophy

**MANDATORY**: All agents must follow Praval's identity-driven design patterns:

```python
@agent("agent_name", responds_to=["message_types"], memory=True)
def agent_function(spore):
    """I am a [clear identity statement]. I specialize in [specific capability]."""
    
    # Extract knowledge from spore
    data = spore.knowledge.get("key")
    
    # Use memory for learning and context
    past_context = agent_function.recall(data, limit=5)
    
    # Apply intelligence through LLM integration
    result = chat(f"Prompt with context: {data}, {past_context}")
    
    # Remember insights for future use
    agent_function.remember(f"Processed {data} -> {result[:100]}...")
    
    # Broadcast results via spore
    broadcast({
        "type": "result_type",
        "knowledge": {"processed_data": result}
    })
```

### Agent Identity Requirements

Each agent MUST have:
- **Clear Identity Statement**: "I am a [role] who specializes in [domain]"
- **Specific Domain**: Well-defined area of expertise
- **Memory Integration**: Uses `remember()` and `recall()` for learning
- **LLM Integration**: Leverages `chat()` for intelligent processing
- **Spore Communication**: Proper knowledge exchange via broadcasts

### Forbidden Patterns

**NEVER** implement agents as:
- Traditional Python classes with methods
- Stateful objects with instance variables
- Complex inheritance hierarchies
- Direct function calls between agents
- Manual message routing or handling

## Test-Driven Development Requirements

### TDD Cycle for Praval Agents

1. **Red**: Write failing tests for agent behavior
2. **Green**: Implement minimal agent to pass tests
3. **Refactor**: Improve agent while maintaining test coverage
4. **Expand**: Add memory, LLM integration, and advanced capabilities

### Agent Testing Standards

**Spore-Based Testing**:
```python
from praval.testing import create_test_spore, capture_broadcasts

def test_paper_discovery_agent():
    # Arrange: Create test spore
    search_spore = create_test_spore({
        "type": "search_request",
        "knowledge": {"query": "machine learning"}
    })
    
    # Act: Execute agent with mocked dependencies
    with capture_broadcasts() as broadcasts:
        with patch('agent_module.external_service') as mock_service:
            mock_service.return_value = expected_result
            agent_function(search_spore)
    
    # Assert: Verify behavior and broadcasts
    assert len(broadcasts) == 1
    assert broadcasts[0]["type"] == "expected_response"
    assert broadcasts[0]["knowledge"]["result"] == expected_value
```

### Test Coverage Requirements

- **Unit Tests**: Individual agent behavior (90%+ coverage)
- **Integration Tests**: Agent communication workflows (85%+ coverage)
- **Memory Tests**: Agent learning and recall functionality
- **Spore Tests**: Message structure and knowledge transfer
- **E2E Tests**: Complete research workflows
- **Performance Tests**: Response times and throughput

### Test Categories

```python
@pytest.mark.unit
def test_agent_core_logic():
    """Test individual agent processing logic"""
    pass

@pytest.mark.integration  
def test_agent_communication():
    """Test agent-to-agent message flows"""
    pass

@pytest.mark.memory
def test_agent_learning():
    """Test memory persistence and recall"""
    pass

@pytest.mark.e2e
def test_research_workflow():
    """Test complete user journeys"""
    pass

@pytest.mark.performance
def test_system_throughput():
    """Test system performance under load"""
    pass
```

## Project Structure Standards

### Required Directory Structure

```
agentic_deep_research/
├── src/
│   ├── agents/                    # Praval agent implementations
│   │   ├── __init__.py
│   │   ├── research/              # Research domain agents
│   │   │   ├── paper_discovery.py
│   │   │   ├── document_processor.py
│   │   │   ├── semantic_analyzer.py
│   │   │   └── summarization.py
│   │   ├── interaction/           # User interaction agents
│   │   │   ├── qa_specialist.py
│   │   │   └── research_advisor.py
│   │   └── coordination/          # System coordination agents
│   │       ├── workflow_coordinator.py
│   │       └── resource_manager.py
│   ├── core/                      # Core infrastructure
│   │   ├── __init__.py
│   │   ├── messaging.py           # RabbitMQ-Praval bridge
│   │   ├── storage.py             # Qdrant/MinIO clients
│   │   └── config.py              # Configuration management
│   ├── api/                       # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   └── models/
│   └── processors/                # Supporting processors
│       ├── __init__.py
│       ├── arxiv_client.py
│       ├── pdf_processor.py
│       └── embeddings.py
├── frontend/                      # React/TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── types/
│   ├── package.json
│   └── tsconfig.json
├── tests/                         # Comprehensive test suite
│   ├── unit/                      # Unit tests for agents
│   ├── integration/               # Integration workflows
│   ├── e2e/                       # End-to-end scenarios
│   ├── performance/               # Load and performance
│   └── fixtures/                  # Test data and mocks
├── config/                        # Configuration files
│   ├── praval.yaml
│   ├── agents.yaml
│   └── infrastructure.yaml
├── docker/                        # Container configurations
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.agents
│   └── Dockerfile.frontend
├── scripts/                       # Development and deployment
│   ├── setup.sh
│   ├── test.sh
│   ├── deploy.sh
│   └── format.sh
├── docs/                          # Documentation
│   ├── architecture.md
│   ├── agent-design.md
│   └── api-reference.md
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── pyproject.toml                 # Project metadata
├── DESIGN.md                      # System design document
├── CLAUDE.md                      # This file
└── README.md                      # Project overview
```

### Code Organization Rules

1. **Agent Modules**: One agent per file, clear domain separation
2. **Core Infrastructure**: Shared utilities, no business logic
3. **API Layer**: Thin controller layer, delegates to agents
4. **Frontend**: Modern React with TypeScript, proper state management
5. **Tests**: Mirror source structure, comprehensive coverage

## Development Standards

### Python Code Quality (Production-Grade)

**Type Hints (MANDATORY)**:
```python
from typing import Dict, Any, List, Optional
from praval.types import Spore

@agent("research_agent", responds_to=["search_request"], memory=True)
def research_agent(spore: Spore) -> Optional[Dict[str, Any]]:
    """I am a research specialist who discovers and analyzes academic papers."""
    query: str = spore.knowledge.get("query", "")
    filters: Dict[str, Any] = spore.knowledge.get("filters", {})
    
    # Implementation with proper typing
    results: List[Dict[str, Any]] = perform_search(query, filters)
    return {"results": results}
```

**Error Handling**:
```python
@agent("document_processor", responds_to=["papers_found"], memory=True)  
def document_processor(spore: Spore) -> None:
    """I process and store research documents with intelligent extraction."""
    try:
        papers = spore.knowledge.get("papers", [])
        
        for paper in papers:
            try:
                process_single_paper(paper)
                document_processor.remember(f"Successfully processed: {paper['title']}")
            except ProcessingError as e:
                logger.error("Paper processing failed", paper_id=paper.get('id'), error=str(e))
                document_processor.remember(f"Failed processing: {paper.get('title')} - {str(e)}")
                continue
                
    except Exception as e:
        logger.error("Critical document processing error", error=str(e))
        broadcast({
            "type": "processing_error",
            "knowledge": {"error": str(e), "context": "document_processing"}
        })
```

**Memory Integration Patterns**:
```python
@agent("query_optimizer", responds_to=["search_optimization_request"], memory=True)
def query_optimizer(spore: Spore) -> None:
    """I optimize research queries using domain knowledge and past success patterns."""
    
    original_query = spore.knowledge.get("query")
    domain = spore.knowledge.get("domain", "general")
    
    # Recall domain-specific patterns
    domain_patterns = query_optimizer.recall(f"domain:{domain}", limit=5)
    successful_queries = query_optimizer.recall("successful_optimization", limit=10)
    
    # Use LLM with memory context
    optimization_prompt = f"""
    Optimize this research query based on successful patterns:
    
    Original: {original_query}
    Domain: {domain}
    Successful patterns: {domain_patterns}
    Historical successes: {successful_queries}
    
    Provide an enhanced query for better research results.
    """
    
    optimized_query = chat(optimization_prompt)
    
    # Remember successful optimization for learning
    query_optimizer.remember(f"successful_optimization: {original_query} -> {optimized_query}")
    query_optimizer.remember(f"domain:{domain} -> pattern: {optimized_query}")
    
    broadcast({
        "type": "query_optimized",
        "knowledge": {
            "original_query": original_query,
            "optimized_query": optimized_query,
            "domain": domain,
            "used_patterns": bool(domain_patterns)
        }
    })
```

### Frontend Development Standards

**React Component Architecture**:
```typescript
// src/components/ResearchInterface.tsx
import React, { useState, useEffect } from 'react';
import { useResearchAgent } from '../hooks/useResearchAgent';
import { ResearchQuery, ResearchResult } from '../types/research';

interface ResearchInterfaceProps {
  onResultsUpdate: (results: ResearchResult[]) => void;
}

export const ResearchInterface: React.FC<ResearchInterfaceProps> = ({ 
  onResultsUpdate 
}) => {
  const [query, setQuery] = useState<string>('');
  const [isSearching, setIsSearching] = useState<boolean>(false);
  
  const { searchPapers, isLoading, error } = useResearchAgent();
  
  const handleSearch = async (): Promise<void> => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    try {
      const results = await searchPapers({
        query: query.trim(),
        maxResults: 10,
        filters: {}
      });
      onResultsUpdate(results);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };
  
  return (
    <div className="research-interface">
      <div className="search-controls">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter research query..."
          className="search-input"
          disabled={isLoading || isSearching}
        />
        <button
          onClick={handleSearch}
          disabled={isLoading || isSearching || !query.trim()}
          className="search-button"
        >
          {isSearching ? 'Searching...' : 'Search Papers'}
        </button>
      </div>
      
      {error && (
        <div className="error-message">
          Error: {error.message}
        </div>
      )}
    </div>
  );
};
```

**TypeScript Type Definitions**:
```typescript
// src/types/research.ts
export interface ResearchQuery {
  query: string;
  maxResults: number;
  filters: ResearchFilters;
  domain?: string;
}

export interface ResearchFilters {
  category?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  authors?: string[];
}

export interface ResearchResult {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  pdfUrl: string;
  publishedDate: string;
  categories: string[];
  relevanceScore: number;
}

export interface AgentResponse<T = any> {
  type: string;
  knowledge: T;
  timestamp: string;
  fromAgent: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: ResearchSource[];
}

export interface ResearchSource {
  title: string;
  paperId: string;
  chunkIndex: number;
  relevanceScore: number;
}
```

## Infrastructure and Deployment

### Development Environment Setup

**Virtual Environment (MANDATORY)**:
```bash
# Create isolated environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with exact versions
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify Praval framework availability
python -c "import praval; print('Praval loaded successfully')"
```

**Environment Variables**:
```env
# Core Application
APP_NAME=agentic_deep_research
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Praval Framework
PRAVAL_DEFAULT_PROVIDER=openai
PRAVAL_DEFAULT_MODEL=gpt-4o-mini
PRAVAL_MAX_THREADS=10
PRAVAL_MEMORY_ENABLED=true
PRAVAL_REEF_CAPACITY=1000

# External Services
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Infrastructure
RABBITMQ_URL=amqp://user:pass@localhost:5672/
QDRANT_URL=http://localhost:6333
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
REDIS_URL=redis://localhost:6379

# ArXiv Configuration  
ARXIV_MAX_RESULTS=50
ARXIV_RATE_LIMIT=3
ARXIV_BASE_URL=http://export.arxiv.org/api/query
```

### Docker Infrastructure

**docker-compose.yml** (Complete Stack):
```yaml
version: '3.8'

services:
  # Message Broker
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: research_user
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: rabbitmq-diagnostics -q ping
      interval: 30s
      timeout: 10s
      retries: 3

  # Vector Database
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: curl -f http://localhost:6333/health
      interval: 30s
      timeout: 10s
      retries: 3

  # Object Storage
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY}
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: curl -f http://localhost:9000/minio/health/live
      interval: 30s
      timeout: 10s
      retries: 3

  # Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: redis-cli ping
      interval: 30s
      timeout: 10s
      retries: 3

  # Praval Agent System
  research_agents:
    build:
      context: .
      dockerfile: docker/Dockerfile.agents
    environment:
      - RABBITMQ_URL=amqp://research_user:${RABBITMQ_PASSWORD}@rabbitmq:5672/
      - QDRANT_URL=http://qdrant:6333
      - MINIO_ENDPOINT=minio:9000
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      rabbitmq:
        condition: service_healthy
      qdrant:
        condition: service_healthy
      minio:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  # FastAPI Backend
  research_api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - RABBITMQ_URL=amqp://research_user:${RABBITMQ_PASSWORD}@rabbitmq:5672/
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
    depends_on:
      - research_agents
    healthcheck:
      test: curl -f http://localhost:8000/health
      interval: 30s
      timeout: 10s
      retries: 3

  # React Frontend
  research_frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - research_api

volumes:
  rabbitmq_data:
  qdrant_data:
  minio_data:
  redis_data:
```

## Testing Excellence

### Pre-commit Hooks ✅ IMPLEMENTED

**Status**: Fully implemented in `.pre-commit-config.yaml`

The project now has automated pre-commit hooks that run on every commit. Install with:
```bash
pre-commit install
```

**Configuration** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/ruff
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### Test Automation

**scripts/test.sh**:
```bash
#!/bin/bash
set -e

echo "🧪 Running comprehensive test suite..."

# Unit tests with coverage
echo "Running unit tests..."
pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

# Integration tests
echo "Running integration tests..."
pytest tests/integration/ -v

# Memory tests for agents
echo "Testing agent memory functionality..."
pytest tests/memory/ -v -m memory

# End-to-end tests
echo "Running E2E tests..."
pytest tests/e2e/ -v -m e2e

# Performance benchmarks
echo "Running performance tests..."
pytest tests/performance/ -v -m performance

# Type checking
echo "Running type checks..."
mypy src/

# Code quality
echo "Running code quality checks..."
ruff check src/
black --check src/

# Security scanning
echo "Running security scan..."
bandit -r src/

echo "✅ All tests passed!"
```

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests for individual components
    integration: Integration tests for component interaction
    memory: Agent memory functionality tests
    e2e: End-to-end workflow tests
    performance: Performance and load tests
    slow: Tests that take significant time to run
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --maxfail=10
    --durations=10
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

## Quality Assurance

### Code Quality Metrics

**Minimum Requirements**:
- **Test Coverage**: 90% for agents, 85% for infrastructure
- **Type Coverage**: 95% type annotation coverage
- **Code Complexity**: Cyclomatic complexity < 10 per function
- **Documentation**: 100% docstring coverage for public APIs
- **Performance**: < 200ms response time for Q&A queries
- **Memory**: Agent recall operations < 100ms

### Continuous Integration ✅ IMPLEMENTED

**Status**: Fully implemented in `.github/workflows/ci.yml`

The project has a comprehensive CI/CD pipeline that runs on all PRs to main:
- Code quality checks (ruff, black, mypy, bandit)
- Test suite with coverage reporting
- Multi-Python version testing (3.9, 3.10, 3.11)
- Service integration (RabbitMQ, Qdrant)

See `DEVELOPMENT.md` for complete developer guide and `.github/workflows/ci.yml` for workflow details.

**Workflow Configuration** (`.github/workflows/ci.yml`):
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    services:
      rabbitmq:
        image: rabbitmq:3-management-alpine
        ports:
          - 5672:5672
        options: --health-cmd "rabbitmq-diagnostics ping" --health-interval 10s
      
      qdrant:
        image: qdrant/qdrant:v1.7.0
        ports:
          - 6333:6333
        options: --health-cmd "curl -f http://localhost:6333/health" --health-interval 10s

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        ruff check src/
        black --check src/
        mypy src/
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Agent-Specific Guidelines

### Research Domain Agents

**Paper Discovery Agent**:
```python
@agent("paper_searcher", responds_to=["search_request"], memory=True)
def paper_discovery_agent(spore: Spore) -> None:
    """I am a research paper discovery specialist. I excel at finding relevant 
    academic papers by understanding research contexts and optimizing queries 
    based on past successful searches."""
    
    query = spore.knowledge.get("query")
    domain = spore.knowledge.get("domain", "general")
    
    # Memory-driven query optimization
    past_searches = paper_discovery_agent.recall(f"domain:{domain}", limit=5)
    successful_patterns = paper_discovery_agent.recall("successful_query", limit=10)
    
    if past_searches or successful_patterns:
        optimization_context = {
            "domain_history": past_searches,
            "successful_patterns": successful_patterns,
            "original_query": query
        }
        
        optimized_query = chat(f"""
        As a research query optimization expert, enhance this query:
        
        Original: {query}
        Domain: {domain}
        Historical context: {optimization_context}
        
        Provide an optimized query that will find more relevant papers.
        Consider synonyms, related terms, and field-specific terminology.
        Return only the enhanced query.
        """)
    else:
        optimized_query = query
    
    # Execute search with error handling
    try:
        papers = arxiv_client.search(
            query=optimized_query.strip(),
            max_results=spore.knowledge.get("max_results", 10)
        )
        
        # Learn from successful searches
        if papers:
            paper_discovery_agent.remember(f"successful_query: {query} -> {len(papers)} papers")
            paper_discovery_agent.remember(f"domain:{domain} -> optimized: {optimized_query}")
        
        broadcast({
            "type": "papers_found",
            "knowledge": {
                "papers": papers,
                "original_query": query,
                "optimized_query": optimized_query,
                "search_metadata": {
                    "domain": domain,
                    "optimization_used": optimized_query != query,
                    "results_count": len(papers)
                }
            }
        })
        
    except Exception as e:
        logger.error("Paper search failed", query=query, error=str(e))
        paper_discovery_agent.remember(f"failed_query: {query} - {str(e)}")
        
        broadcast({
            "type": "search_error",
            "knowledge": {
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__
            }
        })
```

### Interaction Domain Agents

**Q&A Specialist Agent**:
```python
@agent("qa_specialist", responds_to=["user_query"], memory=True)
def qa_specialist_agent(spore: Spore) -> Dict[str, Any]:
    """I am a research Q&A expert who provides comprehensive, accurate answers
    about research papers using retrieved context and accumulated knowledge
    to deliver personalized, insightful responses."""
    
    user_query = spore.knowledge.get("query")
    user_id = spore.knowledge.get("user_id", "anonymous")
    conversation_context = spore.knowledge.get("conversation_context", [])
    
    # Retrieve relevant research context
    relevant_chunks = qdrant_client.similarity_search(user_query, top_k=5)
    
    # Personalization through memory
    user_interests = qa_specialist_agent.recall(f"user:{user_id}", limit=10)
    similar_questions = qa_specialist_agent.recall(user_query, limit=5)
    
    # Build comprehensive context
    context_pieces = []
    sources = []
    
    for chunk in relevant_chunks:
        context_pieces.append({
            "content": chunk.text,
            "source": chunk.metadata.get('title', 'Unknown'),
            "relevance": chunk.score
        })
        sources.append({
            "title": chunk.metadata.get('title'),
            "paper_id": chunk.metadata.get('paper_id'),
            "chunk_index": chunk.metadata.get('chunk_index')
        })
    
    # Generate personalized response
    qa_prompt = f"""
    Answer this research question with expertise and precision:
    
    Question: {user_query}
    
    User's Research Profile:
    {user_interests if user_interests else "No prior interaction history"}
    
    Conversation Context:
    {conversation_context if conversation_context else "Fresh conversation"}
    
    Relevant Research Evidence:
    {chr(10).join([f"From '{c['source']}' (relevance: {c['relevance']:.2f}): {c['content'][:500]}..." for c in context_pieces[:3]])}
    
    Similar Past Questions:
    {similar_questions if similar_questions else "No similar questions found"}
    
    Provide a comprehensive answer that:
    1. Directly addresses the question with specific evidence
    2. Considers the user's apparent research interests
    3. Acknowledges any limitations or uncertainties
    4. Suggests related questions for deeper exploration  
    5. Cites specific papers and findings
    6. Maintains academic rigor while being accessible
    
    Format as a structured response with clear citations.
    """
    
    comprehensive_answer = chat(qa_prompt)
    
    # Generate insightful follow-ups
    followup_prompt = f"""
    Based on this research Q&A:
    
    Question: {user_query}
    Answer: {comprehensive_answer}
    Context: {[c['source'] for c in context_pieces[:3]]}
    
    Suggest 3 specific, insightful follow-up questions that would:
    1. Deepen understanding of the topic
    2. Explore related research areas
    3. Connect to practical applications
    
    Make them research-oriented and thought-provoking.
    """
    
    followup_questions = chat(followup_prompt)
    
    # Remember interaction for personalization
    interaction_memory = f"user:{user_id} asked '{user_query}' -> answered using {len(sources)} sources from {set(s['title'] for s in sources if s['title'])}"
    qa_specialist_agent.remember(interaction_memory)
    
    # Track research interests
    qa_specialist_agent.remember(f"user:{user_id} interest: {user_query}")
    
    response = {
        "type": "qa_response",
        "knowledge": {
            "user_query": user_query,
            "comprehensive_answer": comprehensive_answer,
            "followup_questions": followup_questions,
            "sources": sources,
            "response_metadata": {
                "context_chunks_used": len(relevant_chunks),
                "personalization_applied": bool(user_interests),
                "conversation_length": len(conversation_context)
            }
        }
    }
    
    broadcast(response)
    return response  # For direct API access
```

## Monitoring and Observability

### Agent Performance Monitoring

```python
# src/core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import structlog
from functools import wraps
import time

# Metrics for agent performance
AGENT_MESSAGES_PROCESSED = Counter(
    'agent_messages_processed_total', 
    'Total messages processed by agents',
    ['agent_name', 'message_type', 'status']
)

AGENT_PROCESSING_TIME = Histogram(
    'agent_processing_seconds',
    'Time spent processing messages',
    ['agent_name', 'message_type']
)

AGENT_MEMORY_OPERATIONS = Counter(
    'agent_memory_operations_total',
    'Agent memory operations',
    ['agent_name', 'operation']
)

def monitor_agent_performance(agent_name: str):
    """Decorator to monitor agent performance metrics"""
    def decorator(agent_func):
        @wraps(agent_func)
        def wrapper(spore, *args, **kwargs):
            message_type = spore.get('type', 'unknown')
            start_time = time.time()
            
            try:
                # Execute agent
                result = agent_func(spore, *args, **kwargs)
                
                # Record success metrics
                AGENT_MESSAGES_PROCESSED.labels(
                    agent_name=agent_name,
                    message_type=message_type,
                    status='success'
                ).inc()
                
                return result
                
            except Exception as e:
                # Record failure metrics
                AGENT_MESSAGES_PROCESSED.labels(
                    agent_name=agent_name,
                    message_type=message_type,
                    status='error'
                ).inc()
                
                structlog.get_logger().error(
                    "Agent processing error",
                    agent=agent_name,
                    message_type=message_type,
                    error=str(e)
                )
                raise
                
            finally:
                # Record processing time
                processing_time = time.time() - start_time
                AGENT_PROCESSING_TIME.labels(
                    agent_name=agent_name,
                    message_type=message_type
                ).observe(processing_time)
        
        return wrapper
    return decorator

def monitor_memory_operation(agent_name: str, operation: str):
    """Record agent memory operations"""
    AGENT_MEMORY_OPERATIONS.labels(
        agent_name=agent_name,
        operation=operation
    ).inc()
```

## Success Criteria

### Definition of Done

A feature is complete when:

1. **Agent Implementation**: Proper Praval agent with identity, memory, and LLM integration
2. **Test Coverage**: 90%+ unit test coverage, all integration tests passing
3. **Type Safety**: 95%+ type annotation coverage, mypy passing
4. **Performance**: Meets response time requirements under load
5. **Memory Integration**: Agent learns and improves from interactions
6. **Documentation**: Complete docstrings and usage examples
7. **Security**: No secrets in code, proper input validation
8. **Monitoring**: Metrics and logging in place

### Quality Gates

**Pre-merge Requirements**:
- [ ] All tests passing (unit, integration, e2e)
- [ ] Code coverage above thresholds
- [ ] Type checking clean
- [ ] Performance benchmarks met
- [ ] Security scan clean  
- [ ] Documentation updated
- [ ] Agent identity clearly defined
- [ ] Memory integration tested

**Production Deployment Requirements**:
- [ ] Load testing completed
- [ ] Monitoring dashboards configured
- [ ] Error tracking enabled
- [ ] Backup procedures tested
- [ ] Rollback plan validated
- [ ] Agent memory persisted
- [ ] Multi-instance agent scaling verified

## Knowledge Base Management

### Overview

The system includes a comprehensive Knowledge Base Management interface for viewing, managing, and maintaining the vector store of indexed research papers. This feature provides direct visibility and control over what papers are available for Q&A.

### Architecture Decision: Direct API vs Agent-Based

**Design Philosophy**:
- **Simple CRUD operations** (list, view, delete) use direct API → Qdrant access
- **Intelligent operations** (optimization, curation, recommendations) can be implemented as agents in the future
- This hybrid approach balances pragmatism with Praval's agent-first philosophy

**Rationale**:
Knowledge base viewing and deletion are administrative data management tasks that don't require LLM intelligence or memory. They need fast, synchronous responses. However, intelligent features like "suggest papers to remove based on usage" or "identify redundant papers" would benefit from agent implementation.

### Implementation

#### Backend Components

**1. Qdrant Client Extensions** (`src/agentic_research/storage/qdrant_client.py`):

```python
def get_all_papers(self) -> List[Dict[str, Any]]:
    """
    Get list of all unique papers in the collection with metadata.

    Uses Qdrant scroll API to efficiently retrieve all points, groups by
    paper_id, and returns aggregated metadata for each unique paper.

    Returns:
        List of papers with: paper_id, title, authors, categories,
        published_date, chunk_count, abstract
    """
    # Scroll through all points in batches
    # Group by paper_id
    # Sort by title
    # Return aggregated metadata

def clear_collection(self) -> None:
    """
    Clear all data from the collection by deleting and recreating it.

    This is a destructive operation that removes all papers and vectors.
    The collection is immediately recreated with the same configuration.
    """
    # Delete collection
    # Recreate with same vector configuration
```

**2. REST API Endpoints** (`src/agentic_research/api/routes/research.py`):

```python
@router.get("/knowledge-base/papers")
async def list_indexed_papers() -> Dict[str, Any]:
    """List all papers currently indexed in the knowledge base."""

@router.get("/knowledge-base/stats")
async def get_kb_stats() -> Dict[str, Any]:
    """Get statistics: total papers, vectors, avg chunks, top categories."""

@router.delete("/knowledge-base/papers/{paper_id}")
async def delete_paper(paper_id: str) -> Dict[str, Any]:
    """Delete a specific paper and all its vectors."""

@router.delete("/knowledge-base/clear")
async def clear_knowledge_base() -> Dict[str, Any]:
    """Clear the entire knowledge base (destructive, requires confirmation)."""
```

#### Frontend Components

**1. Knowledge Base Page** (`frontend/knowledge-base.html`):
- Statistics cards showing total papers, total vectors, average chunks per paper
- Sortable table displaying all indexed papers with metadata
- Delete button for individual papers
- Clear all button with double confirmation
- Navigation back to main search interface

**2. JavaScript Logic** (`frontend/kb-app.js`):
- Fetches papers and statistics from API
- Renders responsive table with paper information
- Handles single paper deletion with user confirmation
- Handles full knowledge base clear with double confirmation
- Auto-refreshes data after operations
- Proper error handling and user feedback

**3. Navigation** (`frontend/index.html`):
- Added "📚 Knowledge Base" button in main header
- Seamless navigation between search and knowledge base management

### Usage

**Accessing Knowledge Base**:
1. Navigate to http://localhost:3000
2. Click "📚 Knowledge Base" in header
3. View all indexed papers with statistics

**Managing Papers**:
- **View Details**: See paper title, authors, chunk count, categories
- **Delete Paper**: Click 🗑️ button, confirm deletion
- **Clear All**: Use "Clear All Papers" button, requires double confirmation
- **Refresh**: Click 🔄 button to reload current state

**API Testing**:
```bash
# List all papers
curl http://localhost:8000/research/knowledge-base/papers | jq

# Get statistics
curl http://localhost:8000/research/knowledge-base/stats | jq

# Delete specific paper
curl -X DELETE http://localhost:8000/research/knowledge-base/papers/2312.05589v2

# Clear entire knowledge base (careful!)
curl -X DELETE http://localhost:8000/research/knowledge-base/clear
```

### Data Model

**Paper Metadata Structure**:
```json
{
  "paper_id": "2312.05589v2",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "categories": ["cs.AI", "cs.LG"],
  "published_date": "2023-12-09",
  "chunk_count": 50,
  "abstract": "First 300 chars of abstract..."
}
```

**Statistics Response**:
```json
{
  "total_papers": 28,
  "total_vectors": 1641,
  "avg_chunks_per_paper": 58.6,
  "categories": {
    "cs.AI": 15,
    "cs.LG": 6,
    "cs.NE": 5
  }
}
```

### Performance Considerations

**Efficient Retrieval**:
- Uses Qdrant's `scroll` API for batch retrieval (100 points per batch)
- No vectors loaded (only metadata), reducing network transfer
- Grouping and aggregation done in memory after retrieval
- Results cached on frontend until refresh

**Deletion Operations**:
- Single paper deletion uses Qdrant filter-based delete (efficient)
- Clear operation recreates collection (instant, no iteration needed)
- All operations logged for audit trail

### Future Enhancements (Agent-Based)

**Intelligent Knowledge Base Manager Agent** (future implementation):

```python
@agent("kb_curator", responds_to=["kb_optimization_request"], memory=True)
def knowledge_base_curator(spore):
    """I am a knowledge base curator who intelligently manages the research
    corpus based on usage patterns, relevance, and redundancy."""

    # Analyze usage patterns from Q&A agent memory
    usage_stats = kb_curator.recall("paper_usage", limit=100)

    # Identify redundant papers using LLM
    redundancy_analysis = chat(f"""
    Analyze these papers for content overlap and redundancy:
    {papers_metadata}

    Which papers cover duplicate ground? Which add unique value?
    """)

    # Suggest optimizations
    broadcast({
        "type": "kb_optimization_suggestions",
        "knowledge": {
            "underutilized_papers": [...],
            "redundant_papers": [...],
            "high_value_papers": [...],
            "recommended_removals": [...]
        }
    })
```

**Potential Agent-Based Features**:
- Automatic cleanup of unused papers based on Q&A query patterns
- Smart detection of redundant/duplicate papers
- Usage-based recommendations for knowledge base optimization
- Proactive suggestions when knowledge base grows too large
- Automatic paper quality scoring based on citation relevance

### Security & Safety

**Safeguards**:
- Delete operations require explicit user confirmation
- Clear all requires double confirmation
- All destructive operations logged with timestamp
- No cascade deletion of related resources
- Operations are atomic (either succeed completely or fail)

**Access Control** (future):
- Can add authentication middleware to endpoints
- Role-based access for different operations (view vs delete)
- Audit logging for compliance

### Monitoring

**Key Metrics to Track**:
- Total papers indexed over time
- Average chunk count per paper
- Category distribution changes
- Deletion frequency and patterns
- Knowledge base growth rate
- Storage utilization

This project demonstrates excellence in agentic system design using Praval's powerful patterns while maintaining enterprise-grade quality standards throughout. The Knowledge Base Management feature exemplifies the pragmatic balance between direct API access for simple operations and intelligent agent-based approaches for complex, decision-oriented tasks.