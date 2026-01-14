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
  // Thread-based branching fields
  thread_id: number;        // Thread this message belongs to (0 = original)
  position: number;         // Position within the thread (1-indexed)
  // Version navigation fields (computed by backend)
  has_other_versions?: boolean;  // True if other threads have messages at this position
  version_count?: number;        // Total number of versions at this position
  current_version?: number;      // Which version this is (1-indexed for UI)
}

export interface QuestionRequest {
  question: string;
  include_sources?: boolean;
  conversation_id?: string;
  skip_user_message?: boolean;  // Skip saving user message (used when message already saved from edit)
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
export interface ConversationMetadata {
  paper_ids?: string[];      // Paper IDs for "Chat with Papers" context
  paper_titles?: string[];   // Paper titles for display
  source?: string;           // Origin of conversation (e.g., "kb_search")
  scope?: string;            // Search scope (e.g., "primary_plus_related")
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  // Thread-based branching
  active_thread_id: number;  // Currently active thread (0 = original)
  max_thread_id: number;     // Highest thread ID created
  // Paper context for KB search chats
  metadata?: ConversationMetadata;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

// Thread version info types (for navigation UI)
export interface ThreadVersionInfo {
  position: number;           // The message position
  version_count: number;      // Total versions at this position
  current_thread_id: number;  // Currently active thread
  versions: ThreadVersion[];  // All versions at this position
}

export interface ThreadVersion {
  thread_id: number;
  message_id: string;
  content_preview: string;    // First 100 chars of content
  timestamp: string;
  is_active: boolean;         // True if this is the active thread version
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

// Content Generation types (Twitter/X + Blog Posts)
export type ContentFormat = 'twitter' | 'blog';
export type ContentStyle = 'academic' | 'casual' | 'narrative';

export interface ContentGenerationRequest {
  conversation_id: string;
  format: ContentFormat;
  style?: ContentStyle;
  max_tweets?: number;      // Only for twitter format
  include_toc?: boolean;    // Only for blog format
  custom_prompt?: string;   // Custom instructions to steer the output
}

export interface Tweet {
  position: number;
  content: string;
  char_count: number;
  has_citation: boolean;
  citation_url?: string;
}

export interface BlogPost {
  title: string;
  content: string;
  word_count: number;
  references: string[];
}

export interface ContentGenerationResponse {
  format: ContentFormat;
  style: ContentStyle;
  tweets?: Tweet[];
  blog_post?: BlogPost;
  papers_cited: string[];
  generation_time_ms: number;
}

// Proactive Research Insights types
export * from './insights';

// Knowledge Base Hybrid Search types
export interface KBSearchRequest {
  query: string;
  top_k?: number;
  alpha?: number;  // BM25 weight: 1.0=keyword, 0.5=hybrid, 0.0=semantic
  categories?: string[];
  paper_ids?: string[];
}

export interface KBPaperResult {
  paper_id: string;
  title: string;
  authors: string[];
  categories: string[];
  abstract: string;
  combined_score: number;
  bm25_score: number | null;
  bm25_rank: number | null;
  vector_score: number | null;
  vector_rank: number | null;
  matching_chunks: number;
}

export interface KBSearchResponse {
  query: string;
  search_mode: 'keyword' | 'hybrid' | 'semantic';
  alpha: number;
  results: KBPaperResult[];
  total_found: number;
  search_time_ms: number;
}

export interface KBSearchStats {
  total_papers: number;
  total_chunks: number;
  bm25_ready: boolean;
  vector_ready: boolean;
  last_indexed: string | null;
}

export interface StartPaperChatRequest {
  paper_ids: string[];
  initial_question?: string;
}

export interface StartPaperChatResponse {
  conversation_id: string;
  paper_count: number;
  paper_titles: string[];
  redirect_url: string;
}
