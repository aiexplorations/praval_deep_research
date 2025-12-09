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
  APIError
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

  async listPapers(): Promise<{ papers: Paper[] }> {
    const response = await this.client.get<{ papers: Paper[] }>(
      '/research/knowledge-base/papers'
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

  // Branching Operations
  async editMessage(
    conversationId: string,
    messageId: string,
    newContent: string
  ): Promise<{
    message: any;
    branch_id: string;
    branch_index: number;
    sibling_count: number;
    status: string;
  }> {
    const response = await this.client.post(
      `/research/conversations/${conversationId}/messages/${messageId}/edit`,
      { new_content: newContent }
    );
    return response.data;
  }

  async switchBranch(
    conversationId: string,
    params: { branch_id?: string; message_id?: string; direction?: 'left' | 'right' }
  ): Promise<{
    active_branch_id: string | null;
    messages: any[];
    status: string;
  }> {
    const response = await this.client.post(
      `/research/conversations/${conversationId}/switch-branch`,
      params
    );
    return response.data;
  }

  async getBranchesAtMessage(
    conversationId: string,
    messageId: string
  ): Promise<{
    message_id: string;
    branch_count: number;
    branches: Array<{
      branch_id: string | null;
      branch_index: number;
      message_id: string;
      first_message_preview: string;
      timestamp: string;
    }>;
  }> {
    const response = await this.client.get(
      `/research/conversations/${conversationId}/messages/${messageId}/branches`
    );
    return response.data;
  }

  async deleteBranch(conversationId: string, branchId: string): Promise<{ message: string; status: string }> {
    const response = await this.client.delete(
      `/research/conversations/${conversationId}/branches/${branchId}`
    );
    return response.data;
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
}

// Export singleton instance
export const apiClient = new APIClient();

export default apiClient;
