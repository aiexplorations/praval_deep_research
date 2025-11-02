-- PostgreSQL initialization script for research system metadata
-- This script sets up the database schema for storing research metadata

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS research;
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Grant permissions
GRANT USAGE ON SCHEMA research TO research_user;
GRANT USAGE ON SCHEMA agents TO research_user;
GRANT USAGE ON SCHEMA monitoring TO research_user;

-- Research Papers Metadata Table
CREATE TABLE IF NOT EXISTS research.papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    abstract TEXT,
    categories VARCHAR(100)[],
    published_date TIMESTAMP WITH TIME ZONE,
    updated_date TIMESTAMP WITH TIME ZONE,
    pdf_url TEXT,
    pdf_stored_path TEXT,
    processing_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON research.papers (arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_title_trgm ON research.papers USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_papers_categories ON research.papers USING gin (categories);
CREATE INDEX IF NOT EXISTS idx_papers_published_date ON research.papers (published_date DESC);
CREATE INDEX IF NOT EXISTS idx_papers_processing_status ON research.papers (processing_status);
CREATE INDEX IF NOT EXISTS idx_papers_metadata ON research.papers USING gin (metadata);

-- Research Sessions Table
CREATE TABLE IF NOT EXISTS research.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    session_name VARCHAR(200),
    query TEXT NOT NULL,
    papers_found INTEGER DEFAULT 0,
    papers_processed INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON research.sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON research.sessions (status);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON research.sessions (created_at DESC);

-- Agent Memory Table
CREATE TABLE IF NOT EXISTS agents.memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    memory_type VARCHAR(50) NOT NULL DEFAULT 'episodic',
    content TEXT NOT NULL,
    context_hash VARCHAR(64),
    importance_score FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_memory_agent_name ON agents.memory (agent_name);
CREATE INDEX IF NOT EXISTS idx_memory_type ON agents.memory (memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_context_hash ON agents.memory (context_hash);
CREATE INDEX IF NOT EXISTS idx_memory_importance ON agents.memory (importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_content_trgm ON agents.memory USING gin (content gin_trgm_ops);

-- Agent Performance Metrics Table
CREATE TABLE IF NOT EXISTS monitoring.agent_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    labels JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_metrics_agent_timestamp ON monitoring.agent_metrics (agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON monitoring.agent_metrics (metric_name, timestamp DESC);

-- Research Queue Table (for async processing)
CREATE TABLE IF NOT EXISTS research.processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type VARCHAR(50) NOT NULL,
    paper_id UUID REFERENCES research.papers(id),
    priority INTEGER DEFAULT 5,
    status VARCHAR(20) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_queue_status_priority ON research.processing_queue (status, priority DESC, scheduled_for);
CREATE INDEX IF NOT EXISTS idx_queue_task_type ON research.processing_queue (task_type);

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add update triggers
CREATE TRIGGER update_papers_updated_at BEFORE UPDATE ON research.papers 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA research TO research_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agents TO research_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring TO research_user;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA research TO research_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agents TO research_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO research_user;