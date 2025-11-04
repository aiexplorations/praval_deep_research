/**
 * Discover Page - Paper Search & Discovery
 *
 * Search for research papers from ArXiv with advanced filtering options.
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../services/api/client';
import type { Paper, PaperSearchRequest } from '../types';

export default function Discover() {
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('computer_science');
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());
  const [searchResults, setSearchResults] = useState<Paper[]>([]);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: (request: PaperSearchRequest) => apiClient.searchPapers(request),
    onSuccess: (data) => {
      setSearchResults(data.papers);
      setSelectedPapers(new Set()); // Clear selection on new search
      setSuccessMessage(null); // Clear any previous success message
    }
  });

  // Index mutation
  const indexMutation = useMutation({
    mutationFn: (papers: Paper[]) => apiClient.indexPapers(papers),
    onSuccess: (data) => {
      setSuccessMessage(`Successfully indexed ${data.indexed_count} papers! ${data.vectors_stored} vectors stored.`);
      setSelectedPapers(new Set());
      // Auto-hide after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
    }
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    searchMutation.mutate({
      query: query.trim(),
      domain: domain.replace(/_/g, ' '),
      max_results: 10,
      quality_threshold: 0.3
    });
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
    const papersToIndex = searchResults.filter((p) => p.arxiv_id && selectedPapers.has(p.arxiv_id));
    if (papersToIndex.length === 0) {
      return; // Button is disabled when nothing is selected
    }
    indexMutation.mutate(papersToIndex);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Discover Research Papers</h1>
          <p className="text-muted-foreground">
            Search and index papers from ArXiv for semantic Q&A
          </p>
        </div>

        {/* Success Message */}
        {successMessage && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-green-600 text-xl">✓</span>
              <p className="text-green-800 font-medium">{successMessage}</p>
            </div>
            <button
              onClick={() => setSuccessMessage(null)}
              className="text-green-600 hover:text-green-800"
            >
              ✕
            </button>
          </div>
        )}

        {/* Search Form */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter search query (e.g., 'transformer attention mechanisms')"
              className="flex-1 px-4 py-3 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={searchMutation.isPending}
            />

            <select
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              className="px-4 py-3 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={searchMutation.isPending}
            >
              <option value="computer_science">Computer Science</option>
              <option value="mathematics">Mathematics</option>
              <option value="physics">Physics</option>
              <option value="biology">Biology</option>
              <option value="economics">Economics</option>
            </select>

            <button
              type="submit"
              disabled={searchMutation.isPending || !query.trim()}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {searchMutation.isPending ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>

        {/* Error Display */}
        {searchMutation.isError && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-destructive font-medium">Search Error</p>
            <p className="text-sm text-destructive/80">
              {(searchMutation.error as any)?.message || 'Failed to search papers'}
            </p>
          </div>
        )}

        {/* Index Button */}
        {searchResults.length > 0 && (
          <div className="mb-6 flex items-center justify-between p-4 bg-muted rounded-lg">
            <div className="text-sm text-muted-foreground">
              {selectedPapers.size > 0 ? (
                <span className="font-medium text-foreground">
                  {selectedPapers.size} paper{selectedPapers.size > 1 ? 's' : ''} selected
                </span>
              ) : (
                'Select papers to index for Q&A'
              )}
            </div>
            <button
              onClick={handleIndexSelected}
              disabled={selectedPapers.size === 0 || indexMutation.isPending}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {indexMutation.isPending
                ? 'Indexing...'
                : `Index ${selectedPapers.size || ''} Selected`}
            </button>
          </div>
        )}

        {/* Results Grid */}
        {searchResults.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {searchResults.map((paper) => (
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
        ) : !searchMutation.isPending && searchMutation.isSuccess ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg">No papers found matching your query.</p>
            <p className="text-sm mt-2">Try different search terms or adjust the domain.</p>
          </div>
        ) : null}

        {/* Loading State */}
        {searchMutation.isPending && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-border border-t-primary"></div>
            <p className="mt-4 text-muted-foreground">Searching ArXiv papers...</p>
          </div>
        )}
      </div>
    </div>
  );
}
