/**
 * Core TypeScript types for Praval Deep Research
 */

// Paper types
export interface Paper {
  arxiv_id?: string;  // For search results
  paper_id?: string;  // For knowledge base papers
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  published_date: string;
  url?: string;
  venue?: string;
  relevance_score?: number;
  chunk_count?: number;
}

export interface PaperSearchRequest {
  query: string;
  domain?: string;
  max_results?: number;
  quality_threshold?: number;
}

export interface PaperSearchResponse {
  papers: Paper[];
  total_found: number;
  search_time_ms: number;
  optimization_applied: boolean;
}

// Q&A types
export interface Source {
  title: string;
  paper_id: string;
  excerpt: string;
  relevance_score: number;
  chunk_index?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: string;
}

export interface QuestionRequest {
  question: string;
  include_sources?: boolean;
  conversation_id?: string;
}

export interface QuestionResponse {
  answer: string;
  sources: Source[];
  followup_questions: string[];
  response_time_ms: number;
}

// Collections types
export interface Collection {
  id: string;
  name: string;
  description?: string;
  paper_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface CollectionCreate {
  name: string;
  description?: string;
}

// Tags types
export interface Tag {
  name: string;
  count: number;
}

// Conversation types
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

// Knowledge Base types
export interface KnowledgeBaseStats {
  total_papers: number;
  total_vectors: number;
  avg_chunks_per_paper: number;
  categories: Record<string, number>;
}

// API Error types
export interface APIError {
  message: string;
  status: number;
  details?: unknown;
}

// Agent status types
export interface AgentUpdate {
  message: string;
  status: 'processing' | 'complete' | 'error';
  details?: string;
  timestamp: string;
}

// Voice types (for future use)
export interface VoiceProvider {
  name: 'webspeech' | 'openai-whisper' | 'whisper-wasm';
  stt: 'available' | 'unavailable';
  tts: 'available' | 'unavailable';
}

export interface VoiceTranscript {
  text: string;
  confidence: number;
  isFinal: boolean;
}

// UI State types
export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface ErrorState {
  error: string | null;
  timestamp?: string;
}

// Settings types
export interface UserSettings {
  theme: 'light' | 'dark' | 'system';
  chunkSize: number;
  relevanceThreshold: number;
  voiceEnabled: boolean;
  autoSpeak: boolean;
}

// Proactive Research Insights types
export * from './insights';
