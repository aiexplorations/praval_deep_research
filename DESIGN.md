# Praval Deep Research - System Design

## Overview

Praval Deep Research is a local-first, agent-driven research assistant built on the Praval framework. This document explains the system architecture, data flow, and design decisions using diagrams and plain text (no code).

---

## Design Principles

### 1. Local-First Architecture
All data stays on your machine. Papers, embeddings, and metadata are stored in local Docker volumes. The only external dependencies are:
- ArXiv API (for paper metadata and PDFs)
- OpenAI API (for embeddings and LLM responses)

### 2. Agent-Driven Intelligence
The system uses 6 specialized Praval agents that communicate via message passing (spores). Each agent has:
- **Identity**: What it IS (not just what it does)
- **Memory**: Learns from past interactions
- **Autonomy**: Makes decisions independently
- **Intelligence**: LLM-powered reasoning

### 3. Production-Grade Infrastructure
Enterprise-ready components for reliability, scalability, and observability.

---

## System Architecture

### High-Level Component View

```mermaid
graph TB
    subgraph "User Layer"
        Browser[Web Browser]
    end

    subgraph "Frontend Layer"
        Frontend[Nginx + HTML/JS<br/>Port 3000]
    end

    subgraph "API Layer"
        API[FastAPI Backend<br/>Port 8000]
    end

    subgraph "Agent Layer"
        PA[Paper Discovery<br/>Agent]
        DP[Document Processor<br/>Agent]
        SA[Semantic Analyzer<br/>Agent]
        SU[Summarization<br/>Agent]
        QA[Q&A Specialist<br/>Agent]
        RA[Research Advisor<br/>Agent]
    end

    subgraph "Messaging Layer"
        RMQ[RabbitMQ<br/>Message Queue]
    end

    subgraph "Storage Layer"
        Qdrant[Qdrant<br/>Vector DB]
        MinIO[MinIO<br/>Object Storage]
        Redis[Redis<br/>Cache]
    end

    subgraph "External Services"
        ArXiv[ArXiv API]
        OpenAI[OpenAI API]
    end

    Browser --> Frontend
    Frontend --> API
    API --> RMQ
    RMQ --> PA
    RMQ --> DP
    RMQ --> SA
    RMQ --> SU
    RMQ --> QA
    RMQ --> RA

    PA --> ArXiv
    DP --> MinIO
    DP --> OpenAI
    DP --> Qdrant
    QA --> Qdrant
    QA --> OpenAI

    API --> Redis
    API --> Qdrant
    API --> MinIO

    style Browser fill:#e1f5ff
    style Frontend fill:#ffe1f5
    style API fill:#f5ffe1
    style RMQ fill:#fff3e1
    style Qdrant fill:#e1ffe1
    style MinIO fill:#e1ffe1
    style Redis fill:#e1ffe1
    style ArXiv fill:#ffe1e1
    style OpenAI fill:#ffe1e1
```

### Container Architecture

```mermaid
graph LR
    subgraph "Docker Network: research_network"
        subgraph "Frontend Container"
            F[Nginx<br/>research_frontend]
        end

        subgraph "API Container"
            A[FastAPI<br/>research_api]
        end

        subgraph "Agents Container"
            AG[Praval Agents<br/>research_agents]
        end

        subgraph "Infrastructure Containers"
            R[RabbitMQ<br/>research_rabbitmq]
            Q[Qdrant<br/>research_qdrant]
            M[MinIO<br/>research_minio]
            C[Redis<br/>research_redis]
        end
    end

    F -->|HTTP| A
    A -->|AMQP| R
    AG -->|AMQP| R
    A -->|gRPC| Q
    AG -->|gRPC| Q
    A -->|S3 API| M
    AG -->|S3 API| M
    A -->|Redis Protocol| C

    style F fill:#ffe1f5
    style A fill:#f5ffe1
    style AG fill:#e1f5ff
    style R fill:#fff3e1
    style Q fill:#e1ffe1
    style M fill:#e1ffe1
    style C fill:#e1ffe1
```

---

## Data Flow

### Paper Indexing Workflow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant RabbitMQ
    participant PaperDiscovery
    participant DocProcessor
    participant ArXiv
    participant MinIO
    participant OpenAI
    participant Qdrant

    User->>Frontend: Search "transformers"
    Frontend->>API: POST /research/search
    API->>RabbitMQ: Publish search_request
    RabbitMQ->>PaperDiscovery: Deliver spore

    PaperDiscovery->>ArXiv: Query papers
    ArXiv-->>PaperDiscovery: Paper metadata

    PaperDiscovery->>RabbitMQ: Broadcast papers_found
    RabbitMQ->>DocProcessor: Deliver spore

    DocProcessor->>ArXiv: Download PDF
    ArXiv-->>DocProcessor: PDF bytes
    DocProcessor->>MinIO: Store PDF

    DocProcessor->>DocProcessor: Extract text, chunk

    loop For each chunk
        DocProcessor->>OpenAI: Generate embedding
        OpenAI-->>DocProcessor: 1536-dim vector
    end

    DocProcessor->>Qdrant: Upsert vectors + metadata

    DocProcessor->>RabbitMQ: Broadcast documents_processed

    API-->>Frontend: Return search results
    Frontend-->>User: Display papers
```

### Question Answering Workflow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant RabbitMQ
    participant QAAgent
    participant OpenAI
    participant Qdrant

    User->>Frontend: Ask "What are transformers?"
    Frontend->>API: POST /research/ask
    API->>RabbitMQ: Publish qa_request
    RabbitMQ->>QAAgent: Deliver spore

    QAAgent->>OpenAI: Embed query
    OpenAI-->>QAAgent: Query vector

    QAAgent->>Qdrant: Search similar vectors
    Qdrant-->>QAAgent: Top 5 chunks + metadata

    QAAgent->>QAAgent: Build context from chunks

    QAAgent->>OpenAI: Generate answer with context
    OpenAI-->>QAAgent: Answer text

    QAAgent->>OpenAI: Generate follow-up questions
    OpenAI-->>QAAgent: 3 questions

    QAAgent->>QAAgent: Remember interaction

    QAAgent->>RabbitMQ: Broadcast qa_response

    API-->>Frontend: Return answer + sources
    Frontend-->>User: Display answer with citations
```

---

## Agent Architecture

### Praval Agent Structure

```mermaid
graph TB
    subgraph "Paper Discovery Agent"
        PD_Identity["Identity:<br/>I am a research paper discovery specialist"]
        PD_Memory["Memory:<br/>Successful queries, domains searched"]
        PD_LLM["LLM Integration:<br/>Query optimization"]
        PD_Spore["Spore Communication:<br/>papers_found broadcast"]
    end

    subgraph "Document Processor Agent"
        DP_Identity["Identity:<br/>I process and index research documents"]
        DP_Memory["Memory:<br/>Papers processed, errors encountered"]
        DP_LLM["LLM Integration:<br/>Smart chunking decisions"]
        DP_Spore["Spore Communication:<br/>documents_processed broadcast"]
    end

    subgraph "Q&A Specialist Agent"
        QA_Identity["Identity:<br/>I answer research questions with evidence"]
        QA_Memory["Memory:<br/>User preferences, common questions"]
        QA_LLM["LLM Integration:<br/>Answer generation, follow-ups"]
        QA_Spore["Spore Communication:<br/>qa_response broadcast"]
    end

    style PD_Identity fill:#e1f5ff
    style DP_Identity fill:#ffe1f5
    style QA_Identity fill:#f5ffe1
```

### Message Passing (Spores)

```mermaid
graph LR
    subgraph "Spore Structure"
        Type[Type: message_type]
        Knowledge[Knowledge: key-value data]
        Metadata[Metadata: timestamps, IDs]
    end

    PaperAgent[Paper Discovery Agent]
    DocAgent[Document Processor Agent]
    QAAgent[Q&A Agent]

    PaperAgent -->|papers_found spore| DocAgent
    DocAgent -->|documents_processed spore| QAAgent
    QAAgent -->|qa_response spore| API[API Layer]

    style Type fill:#ffe1e1
    style Knowledge fill:#e1ffe1
    style Metadata fill:#e1f5ff
```

---

## Data Models

### Paper Metadata Flow

```mermaid
graph LR
    ArXiv[ArXiv Paper]
    PDF[PDF Document]
    Text[Extracted Text]
    Chunks[Text Chunks]
    Vectors[Embeddings]
    Qdrant[Qdrant Points]

    ArXiv -->|Download| PDF
    PDF -->|Extract| Text
    Text -->|Split| Chunks
    Chunks -->|Embed| Vectors
    Vectors -->|Store| Qdrant

    ArXiv -.->|Metadata| Qdrant

    style ArXiv fill:#ffe1e1
    style Qdrant fill:#e1ffe1
```

### Vector Point Structure in Qdrant

```
Point {
    id: hash(paper_id + chunk_index)
    vector: [1536-dimensional embedding]
    payload: {
        paper_id: "2312.05589v2"
        title: "Paper Title"
        authors: ["Author 1", "Author 2"]
        categories: ["cs.AI", "cs.LG"]
        published_date: "2023-12-09"
        chunk_text: "Full chunk content..."
        chunk_index: 5
        total_chunks: 50
        abstract: "Paper abstract..."
        pdf_path: "papers/2312.05589v2.pdf"
    }
}
```

---

## Storage Architecture

### Data Persistence Strategy

```mermaid
graph TB
    subgraph "Docker Volumes"
        V1[qdrant_data<br/>Vector DB storage]
        V2[minio_data<br/>PDF files]
        V3[rabbitmq_data<br/>Message queue state]
        V4[redis_data<br/>Cache & sessions]
    end

    subgraph "What's Stored Where"
        V1 --> Vectors[Paper embeddings<br/>+ metadata]
        V2 --> PDFs[Raw PDF files<br/>Downloaded papers]
        V3 --> Messages[Unprocessed messages<br/>Agent state]
        V4 --> Cache[API responses<br/>Session data]
    end

    style V1 fill:#e1ffe1
    style V2 fill:#e1ffe1
    style V3 fill:#fff3e1
    style V4 fill:#ffe1e1
```

### Data Retention

- **Vectors in Qdrant**: Persistent until manually deleted
- **PDFs in MinIO**: Persistent, can grow large (plan ~100MB per paper)
- **Messages in RabbitMQ**: Transient, cleared after processing
- **Cache in Redis**: TTL-based expiration (default 1 hour)

---

## API Design

### REST Endpoints

```mermaid
graph TB
    subgraph "Research Endpoints"
        Search[POST /research/search<br/>Find ArXiv papers]
        Ask[POST /research/ask<br/>Answer questions]
        Index[POST /research/index<br/>Process selected papers]
    end

    subgraph "Knowledge Base Endpoints"
        List[GET /kb/papers<br/>List all indexed papers]
        Stats[GET /kb/stats<br/>Get statistics]
        Delete[DELETE /kb/papers/{id}<br/>Remove specific paper]
        Clear[DELETE /kb/clear<br/>Wipe all papers]
    end

    subgraph "System Endpoints"
        Health[GET /health<br/>Service health check]
        Sessions[GET /research/sessions<br/>Active research sessions]
    end

    style Search fill:#e1f5ff
    style Ask fill:#e1f5ff
    style List fill:#ffe1f5
    style Delete fill:#ffe1e1
```

### Real-Time Updates (SSE)

```mermaid
sequenceDiagram
    participant Browser
    participant SSE as SSE Endpoint<br/>/research/sse/events
    participant Agents

    Browser->>SSE: Connect EventSource
    SSE-->>Browser: Connection established

    loop Agent Updates
        Agents->>SSE: Broadcast agent_status
        SSE-->>Browser: Server-Sent Event
        Browser->>Browser: Update UI
    end

    Browser->>SSE: Close connection
```

---

## Scaling Considerations

### Current Deployment (Single Machine)

```
Load: ~100 papers, ~5,000 vectors
Resources:
  - CPU: 4 cores recommended
  - RAM: 8GB minimum
  - Disk: 10GB for data
  - Network: Minimal (local only)
```

### Horizontal Scaling (Future)

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/HAProxy]
    end

    subgraph "API Instances"
        API1[API Instance 1]
        API2[API Instance 2]
        API3[API Instance 3]
    end

    subgraph "Agent Instances"
        AG1[Agent Workers 1-3]
        AG2[Agent Workers 4-6]
        AG3[Agent Workers 7-9]
    end

    subgraph "Shared Infrastructure"
        RMQ[RabbitMQ Cluster]
        Qdrant[Qdrant Cluster]
        MinIO[MinIO Cluster]
        Redis[Redis Cluster]
    end

    LB --> API1
    LB --> API2
    LB --> API3

    API1 --> RMQ
    API2 --> RMQ
    API3 --> RMQ

    AG1 --> RMQ
    AG2 --> RMQ
    AG3 --> RMQ

    API1 --> Qdrant
    API1 --> MinIO
    API1 --> Redis

    AG1 --> Qdrant
    AG1 --> MinIO
```

---

## Security Model

### Current Implementation

```mermaid
graph TB
    subgraph "Network Security"
        N1[All services on private Docker network]
        N2[Only ports 3000 and 8000 exposed]
        N3[No authentication required - local only]
    end

    subgraph "Data Security"
        D1[No data leaves local machine]
        D2[API keys stored in .env file]
        D3[Docker volumes for persistence]
    end

    subgraph "API Security"
        A1[CORS enabled for localhost]
        A2[Input validation on all endpoints]
        A3[Rate limiting - not implemented]
    end

    style N1 fill:#e1ffe1
    style D1 fill:#e1ffe1
    style A2 fill:#e1ffe1
```

### Future Enhancements

- Add authentication (JWT tokens)
- Implement role-based access control (RBAC)
- Add rate limiting per user
- Enable HTTPS for API
- Add audit logging for deletions

---

## Performance Characteristics

### Benchmarks (Typical)

```
Paper Processing:
  - Download PDF: 2-5 seconds
  - Extract text: 5-10 seconds
  - Generate embeddings: 10-20 seconds (50 chunks)
  - Store vectors: 1-2 seconds
  - Total: 30-60 seconds per paper

Question Answering:
  - Embed query: 0.5 seconds
  - Vector search: 0.1 seconds
  - Generate answer: 2-5 seconds
  - Total: 3-6 seconds per question

Knowledge Base Operations:
  - List papers: 0.2 seconds (100 papers)
  - Delete paper: 0.5 seconds
  - Get stats: 0.3 seconds
```

### Bottlenecks

```mermaid
graph LR
    A[OpenAI API Calls] -->|Rate limits| B[Slowest component]
    C[PDF Download] -->|ArXiv bandwidth| D[Variable latency]
    E[Qdrant Insert] -->|Disk I/O| F[Scales with volume]

    style A fill:#ffe1e1
    style C fill:#fff3e1
    style E fill:#e1ffe1
```

---

## Error Handling Strategy

### Agent-Level Error Handling

```mermaid
graph TD
    A[Agent receives spore] --> B{Can process?}
    B -->|Yes| C[Execute task]
    B -->|No| D[Log error]

    C --> E{Success?}
    E -->|Yes| F[Broadcast result]
    E -->|No| G[Remember failure]

    G --> H[Broadcast error spore]
    D --> H

    F --> I[Update memory]
    H --> I

    style E fill:#e1ffe1
    style D fill:#ffe1e1
    style H fill:#ffe1e1
```

### Retry Strategy

- **Network errors**: 3 retries with exponential backoff
- **OpenAI rate limits**: Respect retry-after headers
- **Qdrant errors**: Single retry, then fail gracefully
- **ArXiv errors**: Single retry with 5 second delay

---

## Monitoring & Observability

### Metrics Collection

```mermaid
graph LR
    subgraph "Application Metrics"
        M1[Papers processed]
        M2[Questions answered]
        M3[Agent response times]
        M4[Error rates]
    end

    subgraph "Infrastructure Metrics"
        I1[Container CPU/Memory]
        I2[Disk usage]
        I3[Network throughput]
        I4[Database query times]
    end

    subgraph "Future: Prometheus"
        P[Metrics endpoint /metrics]
    end

    M1 --> P
    M2 --> P
    M3 --> P
    M4 --> P
    I1 --> P
    I2 --> P
    I3 --> P
    I4 --> P
```

### Logging Strategy

```
Structure: JSON logs with structured fields
Levels:
  - DEBUG: Agent internal state
  - INFO: Normal operations
  - WARNING: Degraded performance
  - ERROR: Operation failures

Log rotation: Daily, keep 7 days
```

---

## Deployment Architecture

### Development vs Production

```mermaid
graph TB
    subgraph "Development"
        D1[Hot reload enabled]
        D2[Debug logging]
        D3[Single container instances]
        D4[No resource limits]
    end

    subgraph "Production"
        P1[Optimized images]
        P2[INFO logging only]
        P3[Multiple replicas]
        P4[CPU/Memory limits set]
    end

    style D1 fill:#ffe1e1
    style P1 fill:#e1ffe1
```

---

## Future Architecture Enhancements

### Planned Improvements

1. **Multi-Source Support**
   - Add PubMed, IEEE, Google Scholar connectors
   - Unified paper metadata schema
   - Source-specific processing agents

2. **Intelligent Curation**
   - Knowledge Base Manager Agent
   - Automatic redundancy detection
   - Usage-based paper recommendations

3. **Advanced Features**
   - Citation graph analysis
   - Paper recommendation engine
   - Research workflow automation
   - Collaborative research sessions

4. **Infrastructure**
   - Kubernetes deployment option
   - Multi-tenant support
   - Distributed vector search
   - Advanced caching strategies

---

This design document focuses on **understanding** the system through diagrams and plain text. For implementation details, see the codebase and CLAUDE.md.
