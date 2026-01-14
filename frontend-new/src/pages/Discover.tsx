/**
 * Discover Page - Paper Search & Discovery
 *
 * Search for research papers from ArXiv or Knowledge Base with hybrid search.
 */

import { useState, useRef } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../services/api/client';
import type { Paper, PaperSearchRequest, KBSearchRequest, KBPaperResult } from '../types';
import ResearchInsights from '../components/insights/ResearchInsights';
import IndexingStatus from '../components/IndexingStatus';
import SearchModeToggle from '../components/discover/SearchModeToggle';
import HybridAlphaSlider from '../components/discover/HybridAlphaSlider';
import { useChatStore } from '../store/useChatStore';

export default function Discover() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { setCurrentConversation, clearMessages } = useChatStore();

  // Search state
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('computer_science');
  const [searchMode, setSearchMode] = useState<'arxiv' | 'knowledge_base'>('arxiv');
  const [hybridAlpha, setHybridAlpha] = useState(0.5);

  // Selection state
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());

  // Results state
  const [arxivResults, setArxivResults] = useState<Paper[]>([]);
  const [kbResults, setKbResults] = useState<KBPaperResult[]>([]);
  const [kbSearchMode, setKbSearchMode] = useState<string>('');
  const [kbSearchTime, setKbSearchTime] = useState<number>(0);

  // UI state
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // ArXiv search mutation
  const arxivSearchMutation = useMutation({
    mutationFn: (request: PaperSearchRequest) => apiClient.searchPapers(request),
    onSuccess: (data) => {
      setArxivResults(data.papers);
      setKbResults([]);
      setSelectedPapers(new Set());
      setSuccessMessage(null);
    }
  });

  // KB search mutation
  const kbSearchMutation = useMutation({
    mutationFn: (request: KBSearchRequest) => apiClient.kbSearch(request),
    onSuccess: (data) => {
      setKbResults(data.results);
      setArxivResults([]);
      setSelectedPapers(new Set());
      setSuccessMessage(null);
      setKbSearchMode(data.search_mode);
      setKbSearchTime(data.search_time_ms);
    }
  });

  // Index mutation (for ArXiv results)
  const indexMutation = useMutation({
    mutationFn: (papers: Paper[]) => apiClient.indexPapers(papers),
    onSuccess: (data) => {
      setSuccessMessage(`Successfully indexed ${data.indexed_count} papers! ${data.vectors_stored} vectors stored.`);
      setSelectedPapers(new Set());
      setTimeout(() => setSuccessMessage(null), 5000);
    }
  });

  // Start paper chat mutation (for KB results)
  const startChatMutation = useMutation({
    mutationFn: (paperIds: string[]) => apiClient.startPaperChat({ paper_ids: paperIds }),
    onSuccess: (data) => {
      // Set up chat store for the new conversation
      setCurrentConversation(data.conversation_id);
      clearMessages();
      // Navigate to chat page
      navigate(`/chat?conversation_id=${data.conversation_id}`);
    }
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    if (searchMode === 'arxiv') {
      arxivSearchMutation.mutate({
        query: query.trim(),
        domain: domain.replace(/_/g, ' '),
        max_results: 10,
        quality_threshold: 0.3
      });
    } else {
      kbSearchMutation.mutate({
        query: query.trim(),
        top_k: 20,
        alpha: hybridAlpha
      });
    }
  };

  const togglePaperSelection = (paperId: string) => {
    const newSelection = new Set(selectedPapers);
    if (newSelection.has(paperId)) {
      newSelection.delete(paperId);
    } else {
      newSelection.add(paperId);
    }
    setSelectedPapers(newSelection);
  };

  const handleIndexSelected = () => {
    const papersToIndex = arxivResults.filter((p) => p.arxiv_id && selectedPapers.has(p.arxiv_id));
    if (papersToIndex.length === 0) return;
    indexMutation.mutate(papersToIndex);
  };

  const handleChatWithSelected = () => {
    const paperIds = Array.from(selectedPapers);
    if (paperIds.length === 0) return;
    startChatMutation.mutate(paperIds);
  };

  const handleInsightClick = (topic: string) => {
    setQuery(topic);
    setTimeout(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      searchInputRef.current?.focus();
    }, 50);

    setTimeout(() => {
      if (searchMode === 'arxiv') {
        arxivSearchMutation.mutate({
          query: topic.trim(),
          domain: domain.replace(/_/g, ' '),
          max_results: 10,
          quality_threshold: 0.3
        });
      } else {
        kbSearchMutation.mutate({
          query: topic.trim(),
          top_k: 20,
          alpha: hybridAlpha
        });
      }
    }, 100);
  };

  const isLoading = arxivSearchMutation.isPending || kbSearchMutation.isPending;
  const hasResults = arxivResults.length > 0 || kbResults.length > 0;

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Discover Research Papers</h1>
          <p className="text-muted-foreground">
            {searchMode === 'arxiv'
              ? 'Search and index papers from ArXiv for semantic Q&A'
              : 'Search your indexed knowledge base with hybrid search'}
          </p>
        </div>

        {/* Success Message */}
        {successMessage && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-green-600 text-xl">&#10003;</span>
              <p className="text-green-800 font-medium">{successMessage}</p>
            </div>
            <button
              onClick={() => setSuccessMessage(null)}
              className="text-green-600 hover:text-green-800"
            >
              &#10005;
            </button>
          </div>
        )}

        {/* Search Mode Toggle */}
        <div className="mb-6">
          <SearchModeToggle
            mode={searchMode}
            onChange={(mode) => {
              setSearchMode(mode);
              setSelectedPapers(new Set());
              setArxivResults([]);
              setKbResults([]);
            }}
            disabled={isLoading}
          />
        </div>

        {/* Hybrid Alpha Slider (only for KB mode) */}
        {searchMode === 'knowledge_base' && (
          <div className="mb-6 p-4 bg-muted/50 rounded-lg">
            <HybridAlphaSlider
              alpha={hybridAlpha}
              onChange={setHybridAlpha}
              disabled={isLoading}
            />
          </div>
        )}

        {/* Search Form */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <input
              ref={searchInputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                searchMode === 'arxiv'
                  ? "Enter search query (e.g., 'transformer attention mechanisms')"
                  : "Search your indexed papers (e.g., 'neural networks for NLP')"
              }
              className="flex-1 px-4 py-3 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={isLoading}
            />

            {searchMode === 'arxiv' && (
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                className="px-4 py-3 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
                disabled={isLoading}
              >
                <option value="computer_science">Computer Science</option>
                <option value="mathematics">Mathematics</option>
                <option value="physics">Physics</option>
                <option value="biology">Biology</option>
                <option value="economics">Economics</option>
              </select>
            )}

            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {isLoading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>

        {/* Error Display */}
        {(arxivSearchMutation.isError || kbSearchMutation.isError) && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-destructive font-medium">Search Error</p>
            <p className="text-sm text-destructive/80">
              {(arxivSearchMutation.error as any)?.message ||
                (kbSearchMutation.error as any)?.message ||
                'Failed to search papers'}
            </p>
          </div>
        )}

        {/* Action Bar */}
        {hasResults && (
          <div className="mb-6 flex items-center justify-between p-4 bg-muted rounded-lg">
            <div className="text-sm text-muted-foreground">
              {selectedPapers.size > 0 ? (
                <span className="font-medium text-foreground">
                  {selectedPapers.size} paper{selectedPapers.size > 1 ? 's' : ''} selected
                </span>
              ) : searchMode === 'arxiv' ? (
                'Select papers to index for Q&A'
              ) : (
                <span>
                  Found {kbResults.length} papers ({kbSearchMode} search, {kbSearchTime}ms)
                </span>
              )}
            </div>

            {searchMode === 'arxiv' ? (
              <button
                onClick={handleIndexSelected}
                disabled={selectedPapers.size === 0 || indexMutation.isPending}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                {indexMutation.isPending
                  ? 'Indexing...'
                  : `Index ${selectedPapers.size || ''} Selected`}
              </button>
            ) : (
              <button
                onClick={handleChatWithSelected}
                disabled={selectedPapers.size === 0 || startChatMutation.isPending}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center gap-2"
              >
                {startChatMutation.isPending ? (
                  'Starting Chat...'
                ) : (
                  <>
                    <span>Chat with {selectedPapers.size || ''}</span>
                    <span>&#9654;</span>
                  </>
                )}
              </button>
            )}
          </div>
        )}

        {/* ArXiv Results Grid */}
        {arxivResults.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {arxivResults.map((paper) => (
              <div
                key={paper.arxiv_id}
                className="border border-border rounded-lg p-6 bg-card hover:shadow-md transition-shadow"
              >
                <div className="flex items-start gap-4">
                  <input
                    type="checkbox"
                    checked={paper.arxiv_id ? selectedPapers.has(paper.arxiv_id) : false}
                    onChange={() => paper.arxiv_id && togglePaperSelection(paper.arxiv_id)}
                    disabled={!paper.arxiv_id}
                    className="mt-1 h-4 w-4 rounded border-border disabled:opacity-50"
                  />
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-lg font-semibold leading-tight">
                        {paper.title}
                      </h3>
                      {paper.relevance_score && (
                        <span className="ml-2 px-2 py-1 bg-primary/10 text-primary text-xs font-medium rounded shrink-0">
                          {(paper.relevance_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>

                    <p className="text-sm text-muted-foreground mb-3 line-clamp-3">
                      {paper.abstract}
                    </p>

                    <div className="flex items-center justify-between text-sm">
                      <div className="text-muted-foreground">
                        <span className="font-medium">
                          {paper.authors.slice(0, 3).join(', ')}
                        </span>
                        {paper.authors.length > 3 && (
                          <span> +{paper.authors.length - 3} more</span>
                        )}
                      </div>

                      <div className="flex gap-2">
                        {paper.url && (
                          <a
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-3 py-1 bg-muted text-muted-foreground rounded hover:bg-muted/80 transition-colors"
                          >
                            ArXiv
                          </a>
                        )}
                        {paper.categories && paper.categories[0] && (
                          <span className="px-3 py-1 bg-muted text-muted-foreground rounded text-xs">
                            {paper.categories[0]}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* KB Results Grid */}
        {kbResults.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {kbResults.map((paper) => (
              <div
                key={paper.paper_id}
                className={`border rounded-lg p-6 bg-card hover:shadow-md transition-all ${
                  selectedPapers.has(paper.paper_id)
                    ? 'border-primary ring-2 ring-primary/20'
                    : 'border-border'
                }`}
              >
                <div className="flex items-start gap-4">
                  <input
                    type="checkbox"
                    checked={selectedPapers.has(paper.paper_id)}
                    onChange={() => togglePaperSelection(paper.paper_id)}
                    className="mt-1 h-4 w-4 rounded border-border"
                  />
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-lg font-semibold leading-tight">
                        {paper.title}
                      </h3>
                      <span className="ml-2 px-2 py-1 bg-primary/10 text-primary text-xs font-medium rounded shrink-0">
                        {(paper.combined_score * 100).toFixed(0)}%
                      </span>
                    </div>

                    <p className="text-sm text-muted-foreground mb-3 line-clamp-3">
                      {paper.abstract}
                    </p>

                    {/* Score breakdown */}
                    <div className="flex flex-wrap gap-2 mb-3">
                      {paper.bm25_score !== null && (
                        <span className="text-xs px-2 py-1 bg-amber-100 text-amber-800 rounded">
                          BM25: {paper.bm25_score.toFixed(1)}
                        </span>
                      )}
                      {paper.vector_score !== null && (
                        <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                          Vector: {paper.vector_score.toFixed(3)}
                        </span>
                      )}
                      <span className="text-xs px-2 py-1 bg-muted text-muted-foreground rounded">
                        {paper.matching_chunks} chunks
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <div className="text-muted-foreground">
                        <span className="font-medium">
                          {paper.authors.slice(0, 3).join(', ')}
                        </span>
                        {paper.authors.length > 3 && (
                          <span> +{paper.authors.length - 3} more</span>
                        )}
                      </div>

                      <div className="flex gap-2">
                        {paper.categories && paper.categories[0] && (
                          <span className="px-3 py-1 bg-muted text-muted-foreground rounded text-xs">
                            {paper.categories[0]}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* No Results Message */}
        {!isLoading &&
          (arxivSearchMutation.isSuccess || kbSearchMutation.isSuccess) &&
          !hasResults && (
            <div className="text-center py-12 text-muted-foreground">
              <p className="text-lg">No papers found matching your query.</p>
              <p className="text-sm mt-2">
                {searchMode === 'arxiv'
                  ? 'Try different search terms or adjust the domain.'
                  : 'Try different keywords or adjust the search balance slider.'}
              </p>
            </div>
          )}

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-border border-t-primary"></div>
            <p className="mt-4 text-muted-foreground">
              {searchMode === 'arxiv'
                ? 'Searching ArXiv papers...'
                : 'Searching knowledge base...'}
            </p>
          </div>
        )}

        {/* Research Insights */}
        <div className="mt-12">
          <ResearchInsights onTopicClick={handleInsightClick} />
        </div>
      </div>

      {/* Indexing Status Overlay */}
      <IndexingStatus
        onComplete={() => {
          queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] });
          queryClient.invalidateQueries({ queryKey: ['knowledge-base-stats'] });
          queryClient.invalidateQueries({ queryKey: ['research-insights'] });
        }}
      />
    </div>
  );
}
