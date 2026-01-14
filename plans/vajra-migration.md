# Migration Plan: Vajra Search + Local LLM Support

## Objective

**Part A: Unified Search (Vajra)**
- Replace Qdrant with Vajra's unified search (BM25 + Vector + Hybrid)
- Eliminate Qdrant service dependency
- Remove OpenAI embedding API costs for queries
- Simplify to one search system

**Part B: Configurable LLM Provider**
- Support any OpenAI-compatible LLM endpoint
- Enable fully local operation (Ollama, LM Studio)
- Support cloud providers (OpenAI, Anthropic, Gemini, Grok, Groq, Together)
- Single configuration point for all agent LLM calls

## Current Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PostgreSQL    │     │     Qdrant      │     │     MinIO       │
│  (conversations │     │ (vector search) │     │  (PDF storage)  │
│   + metadata)   │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                               │
                    ┌─────────────────────┐
                    │   Research API      │
                    │  + Praval Agents    │
                    └─────────────────────┘
                               │
                    ┌─────────────────────┐
                    │   Vajra BM25        │
                    │  (keyword search    │
                    │   for KB Search)    │
                    └─────────────────────┘
```

## Target Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PostgreSQL    │     │     Vajra       │     │     MinIO       │
│  (conversations │     │ (unified search)│     │  (PDF storage)  │
│   + metadata)   │     │  - BM25         │     │                 │
└─────────────────┘     │  - Vector       │     └─────────────────┘
         │              │  - Hybrid       │              │
         │              └─────────────────┘              │
         │                      │                        │
         └──────────────────────┼────────────────────────┘
                               │
                    ┌─────────────────────┐
                    │   Research API      │
                    │  + Praval Agents    │
                    └─────────────────────┘
```

## Services to Keep

| Service | Purpose | Changes |
|---------|---------|---------|
| PostgreSQL | Conversations, metadata | None |
| Redis | Caching, insights | None |
| RabbitMQ | Agent messaging | None |
| MinIO | PDF/file storage | None |
| Research API | REST endpoints | Update search calls |
| Research Agents | Praval agents | Update QA specialist |
| Frontend | React UI | None |

## Services to Remove

| Service | Replacement |
|---------|-------------|
| Qdrant | Vajra Vector Search |
| OpenAI Embeddings (for queries) | Vajra local embeddings |

---

# Part B: Configurable LLM Provider

## Supported Providers

Any LLM with OpenAI-compatible API:

| Provider | Type | Endpoint Example |
|----------|------|------------------|
| **Ollama** | Local | `http://localhost:11434/v1` |
| **LM Studio** | Local | `http://localhost:1234/v1` |
| **OpenAI** | Cloud | `https://api.openai.com/v1` |
| **Anthropic** | Cloud | Via adapter or native |
| **Google Gemini** | Cloud | `https://generativelanguage.googleapis.com/v1beta/openai` |
| **Grok (xAI)** | Cloud | `https://api.x.ai/v1` |
| **Groq** | Cloud | `https://api.groq.com/openai/v1` |
| **Together AI** | Cloud | `https://api.together.xyz/v1` |
| **OpenRouter** | Cloud | `https://openrouter.ai/api/v1` |
| **Local vLLM** | Local | `http://localhost:8000/v1` |

## LLM Provider Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLMProvider (Abstract)                       │
├─────────────────────────────────────────────────────────────────┤
│  + chat(messages, model, temperature, max_tokens) -> str        │
│  + complete(prompt, model, temperature, max_tokens) -> str      │
│  + list_models() -> List[str]                                   │
│  + health_check() -> bool                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ OpenAIProvider  │ │ OllamaProvider  │ │ AnthropicProvider│
│ (OpenAI, Groq,  │ │ (local Ollama)  │ │ (Claude API)     │
│  Together, etc) │ │                 │ │                  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## New File: `src/agentic_research/llm/provider.py`

```python
"""
Configurable LLM Provider supporting any OpenAI-compatible endpoint.

Usage:
    from agentic_research.llm import get_llm_provider

    llm = get_llm_provider()
    response = llm.chat([
        {"role": "user", "content": "Hello!"}
    ])
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from openai import OpenAI
import os


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Send chat completion request."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models."""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for any OpenAI-compatible API.

    Works with: OpenAI, Ollama, LM Studio, Groq, Together, vLLM, etc.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "not-needed",  # Local LLMs often don't need keys
        default_model: str = "gpt-4o-mini",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = default_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def list_models(self) -> List[str]:
        models = self.client.models.list()
        return [m.id for m in models.data]


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude API (native, not OpenAI-compatible)."""

    def __init__(self, api_key: str, default_model: str = "claude-3-haiku-20240307"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.default_model = default_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        # Convert OpenAI format to Anthropic format
        system = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append(msg)

        response = self.client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens,
            system=system,
            messages=anthropic_messages,
        )
        return response.content[0].text

    def list_models(self) -> List[str]:
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]


# Factory function
def get_llm_provider() -> LLMProvider:
    """
    Get configured LLM provider based on environment variables.

    Environment Variables:
        LLM_PROVIDER: "openai", "ollama", "lmstudio", "anthropic", "groq", "together"
        LLM_BASE_URL: Custom endpoint URL (for OpenAI-compatible)
        LLM_API_KEY: API key (optional for local)
        LLM_DEFAULT_MODEL: Default model to use
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Preset configurations
    PRESETS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3.2",
        },
        "lmstudio": {
            "base_url": "http://localhost:1234/v1",
            "default_model": "local-model",
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "llama-3.1-70b-versatile",
        },
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "default_model": "meta-llama/Llama-3-70b-chat-hf",
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "default_model": "gemini-1.5-flash",
        },
        "grok": {
            "base_url": "https://api.x.ai/v1",
            "default_model": "grok-beta",
        },
    }

    if provider == "anthropic":
        return AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            default_model=os.getenv("LLM_DEFAULT_MODEL", "claude-3-haiku-20240307"),
        )

    preset = PRESETS.get(provider, PRESETS["openai"])

    return OpenAICompatibleProvider(
        base_url=os.getenv("LLM_BASE_URL", preset["base_url"]),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "not-needed")),
        default_model=os.getenv("LLM_DEFAULT_MODEL", preset["default_model"]),
    )
```

## Praval Integration

Update Praval's `chat()` function to use the configurable provider:

**File:** `src/agentic_research/llm/praval_adapter.py`

```python
"""
Adapter to make Praval agents use our configurable LLM provider.
"""

from praval import set_llm_provider
from agentic_research.llm.provider import get_llm_provider

def configure_praval_llm():
    """
    Configure Praval to use our LLM provider.

    Call this at application startup.
    """
    llm = get_llm_provider()

    # Praval's chat() function will use this
    def praval_chat(prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        if "system" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system"]})
        return llm.chat(
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
        )

    set_llm_provider(praval_chat)
```

## Environment Configuration Examples

**.env for Local (Ollama):**
```bash
# LLM Provider
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_DEFAULT_MODEL=llama3.2
# No API key needed for local

# Search
VAJRA_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**.env for Cloud (OpenAI):**
```bash
# LLM Provider
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_DEFAULT_MODEL=gpt-4o-mini

# Search
VAJRA_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**.env for Cloud (Groq - fast inference):**
```bash
# LLM Provider
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_DEFAULT_MODEL=llama-3.1-70b-versatile

# Search
VAJRA_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**.env for Fully Local:**
```bash
# LLM Provider (Ollama)
LLM_PROVIDER=ollama
LLM_DEFAULT_MODEL=llama3.2:8b

# Embeddings (local via Vajra)
VAJRA_EMBEDDING_MODEL=all-MiniLM-L6-v2

# No external API keys needed!
```

---

# Migration Phases

## Part A: Vajra Search Migration

### Phase 1: Setup Vajra Unified Index

**Branch:** `feature/vajra-migration`

**Tasks:**
1. Create unified Vajra index manager that handles both BM25 and vector
2. Configure embedding model (recommend: `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`)
3. Set up index persistence path in Docker volume

**New File:** `src/agentic_research/storage/vajra_search.py`

```python
from vajra_bm25 import VajraSearchOptimized, HybridSearchEngine
from vajra_bm25.vector import VajraVectorSearch, TextEmbeddingMorphism

class VajraUnifiedSearch:
    """
    Unified search combining BM25 + Vector with configurable hybrid fusion.
    Replaces Qdrant for all search operations.
    """

    def __init__(
        self,
        index_path: str = "/app/data/vajra_indexes",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.embedder = TextEmbeddingMorphism(embedding_model, device=device)
        self.bm25_engine = VajraSearchOptimized()
        self.vector_engine = VajraVectorSearch(embedder=self.embedder)
        self.hybrid_engine = HybridSearchEngine(
            bm25_engine=self.bm25_engine,
            vector_engine=self.vector_engine,
            method="rrf"
        )

    def index_paper(self, paper_id, title, chunks, metadata):
        """Index a paper's chunks for both BM25 and vector search."""
        pass

    def search(self, query, top_k=10, mode="hybrid", alpha=0.5):
        """
        Search with configurable mode.

        Args:
            mode: "bm25", "vector", or "hybrid"
            alpha: BM25 weight for hybrid (1.0=pure BM25, 0.0=pure vector)
        """
        pass
```

### Phase 2: Data Migration Script

**Tasks:**
1. Export existing data from Qdrant
2. Re-index into Vajra (both BM25 and vector indices)
3. Verify document counts match

**New File:** `scripts/migrate_qdrant_to_vajra.py`

```python
"""
Migration script: Qdrant -> Vajra

Usage:
    python scripts/migrate_qdrant_to_vajra.py --verify
"""

def export_from_qdrant():
    """Export all documents from Qdrant collections."""
    pass

def import_to_vajra():
    """Index all documents into Vajra unified search."""
    pass

def verify_migration():
    """Compare document counts and sample searches."""
    pass
```

### Phase 3: Update Search Consumers

**Files to modify:**

| File | Change |
|------|--------|
| `src/agents/interaction/qa_specialist.py` | Replace `QdrantClientWrapper` with `VajraUnifiedSearch` |
| `src/agentic_research/api/routes/research.py` | Replace `get_vector_search_client()` with Vajra |
| `src/agentic_research/api/routes/kb_search.py` | Already uses hybrid, update to new unified class |
| `src/agentic_research/storage/hybrid_search.py` | Refactor to use `VajraUnifiedSearch` |

**QA Specialist Changes:**

```python
# Before (Qdrant)
from agentic_research.storage.qdrant_client import QdrantClientWrapper
qdrant_client = QdrantClientWrapper(settings)
results = qdrant_client.search_similar(query_vector, limit=10)

# After (Vajra)
from agentic_research.storage.vajra_search import get_vajra_search
vajra = get_vajra_search()
results = vajra.search(query, top_k=10, mode="hybrid")
```

### Phase 4: Update Document Indexing Pipeline

**Files to modify:**

| File | Change |
|------|--------|
| `src/agents/research/document_processor.py` | Index to Vajra instead of Qdrant |
| `src/agentic_research/storage/qdrant_client.py` | Deprecate, redirect to Vajra |
| `scripts/init_vajra_indexes.py` | Update to use unified index |

### Phase 5: Remove Qdrant

**Tasks:**
1. Remove Qdrant from `docker-compose.yml`
2. Remove `qdrant-client` from `requirements.txt`
3. Delete `src/agentic_research/storage/qdrant_client.py`
4. Update health checks to not require Qdrant
5. Clean up environment variables

**docker-compose.yml changes:**

```yaml
# Remove this service:
# qdrant:
#   image: qdrant/qdrant:v1.7.3
#   ports:
#     - "6333:6333"
#   volumes:
#     - qdrant_data:/qdrant/storage

# Remove from depends_on in research_api and research_agents
```

### Phase 6: Testing & Validation

**Test Cases:**

1. **Search Quality:**
   - Compare top-10 results for 20 sample queries (Qdrant vs Vajra)
   - Measure recall@10, precision@10
   - Verify hybrid search improves over BM25-only

2. **Performance:**
   - Measure query latency (target: <100ms for BM25, <200ms for hybrid)
   - Measure indexing throughput
   - Memory usage comparison

3. **Functionality:**
   - KB Search page works with all three modes
   - Chat with Papers filters correctly
   - Q&A provides relevant answers
   - Paper indexing works end-to-end

---

## Part B: LLM Provider Migration

### Phase 7: Create LLM Provider Abstraction

**Tasks:**
1. Create `src/agentic_research/llm/provider.py` with abstract provider class
2. Implement `OpenAICompatibleProvider` (works with most providers)
3. Implement `AnthropicProvider` (native Claude API)
4. Create factory function `get_llm_provider()`
5. Add preset configurations for common providers

### Phase 8: Update All LLM Calls

**Files to modify:**

| File | Current | Change |
|------|---------|--------|
| `src/agents/interaction/qa_specialist.py` | `chat()` from Praval | Use `get_llm_provider()` |
| `src/agents/research/*.py` | Various `chat()` calls | Use `get_llm_provider()` |
| `src/agentic_research/api/routes/research.py` | Direct `OpenAI()` calls | Use `get_llm_provider()` |
| `src/agents/content/content_generator.py` | `chat()` from Praval | Use `get_llm_provider()` |

**Pattern for updating agents:**

```python
# Before
from praval import chat
response = chat("Summarize this...")

# After
from agentic_research.llm import get_llm_provider
llm = get_llm_provider()
response = llm.chat([{"role": "user", "content": "Summarize this..."}])
```

### Phase 9: Praval Framework Integration

**Option A: Monkey-patch Praval's chat function**
```python
# At startup
import praval
from agentic_research.llm import get_llm_provider

llm = get_llm_provider()
praval.chat = lambda prompt, **kw: llm.chat([{"role": "user", "content": prompt}], **kw)
```

**Option B: Create wrapper for Praval agents**
- Less invasive, agents explicitly choose provider

### Phase 10: Add Ollama/LM Studio to Docker Compose (Optional)

For fully self-contained local deployment:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

### Phase 11: Testing LLM Providers

**Test matrix:**

| Provider | Test |
|----------|------|
| OpenAI | Q&A, summarization, content generation |
| Ollama (llama3.2) | Same tests, verify quality |
| Groq | Same tests, verify speed |
| Anthropic | Same tests, verify Claude works |

**Quality benchmarks:**
- Run same 20 questions across providers
- Compare answer quality (human eval or LLM-as-judge)
- Measure latency per provider

---

## Configuration

**Environment Variables:**

```bash
# New
VAJRA_INDEX_PATH=/app/data/vajra_indexes
VAJRA_EMBEDDING_MODEL=all-MiniLM-L6-v2
VAJRA_DEVICE=cpu  # or cuda, mps

# Remove
# QDRANT_HOST=qdrant
# QDRANT_PORT=6333
```

**Embedding Model Options:**

| Model | Dimensions | Speed | Quality | Size |
|-------|------------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | 80MB |
| all-mpnet-base-v2 | 768 | Medium | Better | 420MB |
| BAAI/bge-small-en-v1.5 | 384 | Fast | Better | 130MB |
| BAAI/bge-base-en-v1.5 | 768 | Medium | Best | 440MB |

**Recommendation:** Start with `all-MiniLM-L6-v2` for speed, upgrade to `bge-small-en-v1.5` if quality needs improvement.

## Rollback Plan

If issues arise:
1. Keep Qdrant data volume intact during migration
2. Maintain `qdrant_client.py` as deprecated but functional
3. Feature flag: `USE_VAJRA_SEARCH=true/false` to switch between implementations
4. Can restore Qdrant service from docker-compose backup

## Files Summary

**New Files:**
- `src/agentic_research/storage/vajra_search.py` - Unified search class
- `src/agentic_research/llm/__init__.py` - LLM package
- `src/agentic_research/llm/provider.py` - LLM provider abstraction
- `src/agentic_research/llm/praval_adapter.py` - Praval integration
- `scripts/migrate_qdrant_to_vajra.py` - Migration script

**Modified Files:**
- `src/agents/interaction/qa_specialist.py` - Use Vajra + LLM provider
- `src/agents/research/*.py` - Use LLM provider
- `src/agents/content/content_generator.py` - Use LLM provider
- `src/agentic_research/api/routes/research.py` - Use Vajra + LLM provider
- `src/agentic_research/api/routes/kb_search.py` - Use unified Vajra
- `src/agentic_research/storage/hybrid_search.py` - Refactor to VajraUnifiedSearch
- `src/agentic_research/api/main.py` - Initialize LLM provider at startup
- `docker-compose.yml` - Remove Qdrant, optionally add Ollama
- `requirements.txt` - Remove qdrant-client, add anthropic (optional)
- `.env.example` - Add LLM provider config

**Deleted Files:**
- `src/agentic_research/storage/qdrant_client.py`
- `src/agentic_research/storage/embeddings.py` (OpenAI embeddings)

## Estimated Effort

### Part A: Vajra Search

| Phase | Effort |
|-------|--------|
| Phase 1: Setup Vajra Unified Index | 2-3 hours |
| Phase 2: Data Migration Script | 1-2 hours |
| Phase 3: Update Search Consumers | 2-3 hours |
| Phase 4: Update Indexing Pipeline | 1-2 hours |
| Phase 5: Remove Qdrant | 1 hour |
| Phase 6: Testing & Validation | 2-3 hours |
| **Subtotal** | **10-14 hours** |

### Part B: LLM Provider

| Phase | Effort |
|-------|--------|
| Phase 7: Create LLM Provider Abstraction | 2-3 hours |
| Phase 8: Update All LLM Calls | 3-4 hours |
| Phase 9: Praval Framework Integration | 1-2 hours |
| Phase 10: Docker Compose (Optional) | 1 hour |
| Phase 11: Testing LLM Providers | 2-3 hours |
| **Subtotal** | **9-13 hours** |

### Total

| Scope | Effort |
|-------|--------|
| Part A only (Vajra) | 10-14 hours |
| Part B only (LLM) | 9-13 hours |
| **Both Parts** | **19-27 hours** |

## Success Criteria

### Part A: Vajra Search
1. All existing functionality works without Qdrant
2. Search quality maintained or improved (hybrid)
3. Query latency < 200ms (p95)
4. No OpenAI embedding API calls for search queries
5. Docker memory footprint reduced (no Qdrant container)

### Part B: LLM Provider
6. All agents work with configurable LLM provider
7. Ollama/local LLMs produce acceptable quality answers
8. Switching providers requires only env var changes
9. No code changes needed to switch between OpenAI/Ollama/etc.

### Overall
10. Fully local deployment possible (Ollama + Vajra, no external APIs)
11. All tests pass
12. Documentation updated with provider configuration
