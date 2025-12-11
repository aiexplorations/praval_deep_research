/**
 * API Client for Praval Deep Research
 *
 * Centralized API communication layer with error handling,
 * request/response interceptors, and retry logic.
 */

import axios from 'axios';
import type { AxiosInstance, AxiosError } from 'axios';
import type {
  PaperSearchRequest,
  PaperSearchResponse,
  QuestionRequest,
  QuestionResponse,
  Paper,
  Collection,
  CollectionCreate,
  Tag,
  Conversation,
  ConversationWithMessages,
  KnowledgeBaseStats,
  ResearchInsights,
  APIError,
  Message,
  ThreadVersionInfo,
  ContentFormat,
  ContentStyle,
  ContentGenerationResponse
} from '../../types';

const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : '/api';

class APIClient {
  private client: AxiosInstance;

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      timeout: 60000, // 60 seconds
      headers: {
        'Content-Type': 'application/json'
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add any auth tokens here if needed
        // config.headers.Authorization = `Bearer ${token}`;
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const apiError: APIError = {
          message: error.message,
          status: error.response?.status || 500,
          details: error.response?.data
        };

        // Handle specific error cases
        if (error.response?.status === 404) {
          apiError.message = 'Resource not found';
        } else if (error.response?.status === 500) {
          apiError.message = 'Server error. Please try again later.';
        } else if (error.code === 'ECONNABORTED') {
          apiError.message = 'Request timeout. Please try again.';
        }

        return Promise.reject(apiError);
      }
    );
  }

  // Paper Search
  async searchPapers(request: PaperSearchRequest): Promise<PaperSearchResponse> {
    const response = await this.client.post<PaperSearchResponse>(
      '/research/search',
      request
    );
    return response.data;
  }

  // Index Papers
  async indexPapers(papers: Paper[]): Promise<{ indexed_count: number; vectors_stored: number }> {
    const response = await this.client.post('/research/index', { papers });
    return response.data;
  }

  // Q&A
  async askQuestion(request: QuestionRequest): Promise<QuestionResponse> {
    const response = await this.client.post<QuestionResponse>(
      '/research/ask',
      request
    );
    return response.data;
  }

  // Knowledge Base
  async getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
    const response = await this.client.get<KnowledgeBaseStats>(
      '/research/knowledge-base/stats'
    );
    return response.data;
  }

  async listPapers(params?: {
    search?: string;
    category?: string;
    source?: 'all' | 'kb' | 'linked';
    sort?: 'title' | 'date' | 'date_added' | 'chunks';
    sort_order?: 'asc' | 'desc';
    page?: number;
    page_size?: number;
  }): Promise<{
    papers: Paper[];
    total_papers: number;
    total_vectors: number;
    page: number;
    page_size: number;
    total_pages: number;
    available_categories: string[];
    status: string;
  }> {
    const response = await this.client.get('/research/knowledge-base/papers', { params });
    return response.data;
  }

  async getAreaPapers(areaName: string, limit: number = 10): Promise<{
    area_name: string;
    papers: Array<{
      paper_id: string;
      title: string;
      authors: string[];
      abstract: string;
      categories: string[];
      relevance_score: number;
    }>;
    total_found: number;
    status: string;
  }> {
    const response = await this.client.get(
      `/research/areas/${encodeURIComponent(areaName)}/papers`,
      { params: { limit } }
    );
    return response.data;
  }

  async deletePaper(paperId: string): Promise<{ message: string }> {
    const response = await this.client.delete(
      `/research/knowledge-base/papers/${paperId}`
    );
    return response.data;
  }

  async clearKnowledgeBase(): Promise<{ message: string }> {
    const response = await this.client.delete('/research/knowledge-base/clear');
    return response.data;
  }

  getPaperPdfUrl(paperId: string): string {
    // Return direct API endpoint for PDF viewing
    return `/api/research/knowledge-base/papers/${paperId}/pdf`;
  }

  // Related Papers (Citation Extraction)
  async getRelatedPapers(paperId: string): Promise<{
    paper_id: string;
    paper_title: string;
    related_papers: Array<{
      arxiv_id: string;
      title: string;
      authors: string[];
      abstract: string;
      published_date: string | null;
      categories: string[];
      url: string;
      relevance: string;
      source_paper_id: string;
      source_paper_title: string;
      already_indexed: boolean;
    }>;
    citations_extracted: number;
    papers_found: number;
    message: string;
  }> {
    const response = await this.client.get(
      `/research/knowledge-base/papers/${paperId}/related`
    );
    return response.data;
  }

  // Proactive Research Insights
  async getResearchInsights(): Promise<ResearchInsights> {
    const response = await this.client.get<ResearchInsights>('/research/insights');
    return response.data;
  }

  // Collections (to be implemented in backend)
  async listCollections(): Promise<Collection[]> {
    const response = await this.client.get<Collection[]>('/collections');
    return response.data;
  }

  async createCollection(data: CollectionCreate): Promise<Collection> {
    const response = await this.client.post<Collection>('/collections', data);
    return response.data;
  }

  async updateCollection(id: string, data: Partial<CollectionCreate>): Promise<Collection> {
    const response = await this.client.put<Collection>(`/collections/${id}`, data);
    return response.data;
  }

  async deleteCollection(id: string): Promise<{ message: string }> {
    const response = await this.client.delete(`/collections/${id}`);
    return response.data;
  }

  async addPapersToCollection(collectionId: string, paperIds: string[]): Promise<Collection> {
    const response = await this.client.post<Collection>(
      `/collections/${collectionId}/papers`,
      { paper_ids: paperIds }
    );
    return response.data;
  }

  async removePaperFromCollection(collectionId: string, paperId: string): Promise<Collection> {
    const response = await this.client.delete(
      `/collections/${collectionId}/papers/${paperId}`
    );
    return response.data;
  }

  // Tags (to be implemented in backend)
  async listTags(): Promise<Tag[]> {
    const response = await this.client.get<Tag[]>('/tags');
    return response.data;
  }

  async addTagsToPaper(paperId: string, tags: string[]): Promise<{ message: string }> {
    const response = await this.client.post(
      `/papers/${paperId}/tags`,
      { tags }
    );
    return response.data;
  }

  async removeTagFromPaper(paperId: string, tag: string): Promise<{ message: string }> {
    const response = await this.client.delete(`/papers/${paperId}/tags/${tag}`);
    return response.data;
  }

  // Conversations
  async listConversations(): Promise<{ conversations: Conversation[] }> {
    const response = await this.client.get<{ conversations: Conversation[] }>('/research/conversations');
    return response.data;
  }

  async getConversation(id: string): Promise<ConversationWithMessages> {
    const response = await this.client.get<ConversationWithMessages>(
      `/research/conversations/${id}`
    );
    return response.data;
  }

  async createConversation(title?: string): Promise<Conversation> {
    const response = await this.client.post<Conversation>('/research/conversations', { title });
    return response.data;
  }

  async deleteConversation(id: string): Promise<{ message: string }> {
    const response = await this.client.delete(`/research/conversations/${id}`);
    return response.data;
  }

  async updateConversationTitle(id: string, title: string): Promise<{ message: string }> {
    const response = await this.client.put(`/research/conversations/${id}`, { title });
    return response.data;
  }

  /**
   * Find a conversation by exact title match, or create one if not found.
   * Returns the conversation ID and whether it was newly created.
   */
  async findOrCreateConversation(title: string): Promise<{ id: string; created: boolean }> {
    // First, list all conversations to find a matching title
    const { conversations } = await this.listConversations();
    const existing = conversations.find(c => c.title === title);

    if (existing) {
      return { id: existing.id, created: false };
    }

    // Create new conversation with the title
    const newConversation = await this.createConversation(title);
    return { id: newConversation.id, created: true };
  }

  // Thread-Based Branching Operations

  /**
   * Edit a user message by creating a new thread.
   * This copies all messages up to the edit point into a new thread.
   */
  async editMessage(
    conversationId: string,
    messageId: string,
    newContent: string
  ): Promise<{
    message: Message;
    thread_id: number;
    position: number;
    version_count: number;
    current_version: number;
    status: string;
  }> {
    const response = await this.client.post(
      `/research/conversations/${conversationId}/messages/${messageId}/edit`,
      { new_content: newContent }
    );
    return response.data;
  }

  /**
   * Switch to a different thread in the conversation.
   * Can switch directly by thread_id or navigate by position+direction.
   */
  async switchThread(
    conversationId: string,
    params: { thread_id?: number; position?: number; direction?: 'prev' | 'next' }
  ): Promise<{
    active_thread_id: number;
    messages: Message[];
    status: string;
  }> {
    const response = await this.client.post(
      `/research/conversations/${conversationId}/switch-thread`,
      params
    );
    return response.data;
  }

  /**
   * Get information about all thread versions at a specific position.
   * Used for the < 1/3 > navigation UI.
   */
  async getThreadsAtPosition(
    conversationId: string,
    position: number
  ): Promise<ThreadVersionInfo> {
    const response = await this.client.get(
      `/research/conversations/${conversationId}/threads/${position}`
    );
    return response.data;
  }

  /**
   * Delete a thread and all its messages.
   * Cannot delete thread 0 (original conversation).
   */
  async deleteThread(
    conversationId: string,
    threadId: number
  ): Promise<{ message: string; status: string }> {
    const response = await this.client.delete(
      `/research/conversations/${conversationId}/threads/${threadId}`
    );
    return response.data;
  }

  // Deprecated: Use switchThread instead
  async switchBranch(
    conversationId: string,
    params: { thread_id?: number; position?: number; direction?: 'prev' | 'next' }
  ): Promise<{
    active_thread_id: number;
    messages: Message[];
    status: string;
  }> {
    return this.switchThread(conversationId, params);
  }

  async exportConversation(id: string, format: 'markdown' | 'json' | 'pdf'): Promise<Blob> {
    const response = await this.client.post(
      `/conversations/${id}/export`,
      { format },
      { responseType: 'blob' }
    );
    return response.data;
  }

  // Research Advisor (to be implemented in backend)
  async chatWithAdvisor(message: string): Promise<{ response: string }> {
    const response = await this.client.post('/advisor/chat', { message });
    return response.data;
  }

  async generateResearchPlan(topic: string): Promise<{ plan: string }> {
    const response = await this.client.post('/advisor/plan', { topic });
    return response.data;
  }

  // Summarization (to be implemented in backend)
  async summarizePaper(paperId: string, length?: 'brief' | 'detailed' | 'comprehensive'): Promise<{ summary: string }> {
    const response = await this.client.post(`/papers/${paperId}/summarize`, { length });
    return response.data;
  }

  async getPaperSummary(paperId: string): Promise<{ summary: string | null }> {
    const response = await this.client.get(`/papers/${paperId}/summary`);
    return response.data;
  }

  // Content Generation (Twitter/X + Blog Posts)
  async generateContent(
    conversationId: string,
    options: {
      format: ContentFormat;
      style?: ContentStyle;
      maxTweets?: number;
      includeToc?: boolean;
      customPrompt?: string;
    }
  ): Promise<ContentGenerationResponse> {
    const response = await this.client.post<ContentGenerationResponse>(
      `/research/conversations/${conversationId}/generate-content`,
      {
        conversation_id: conversationId,
        format: options.format,
        style: options.style || 'academic',
        max_tweets: options.maxTweets || 10,
        include_toc: options.includeToc !== false,
        custom_prompt: options.customPrompt || ''
      }
    );
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new APIClient();

export default apiClient;
