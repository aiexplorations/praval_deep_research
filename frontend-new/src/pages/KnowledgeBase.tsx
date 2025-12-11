/**
 * Knowledge Base Page - Manage Indexed Papers
 *
 * View, search, and manage all indexed papers in the knowledge base.
 * Supports filtering by search, category, source, and sorting options.
 */

import { useState, useCallback, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../services/api/client';
import type { Paper } from '../types';
import RelatedPapersModal from '../components/RelatedPapersModal';

interface RelatedPaper {
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
}

type SortField = 'title' | 'date' | 'date_added' | 'chunks';
type SortOrder = 'asc' | 'desc';
type SourceFilter = 'all' | 'kb' | 'linked';

export default function KnowledgeBase() {
  // Filter state
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [category, setCategory] = useState<string>('');
  const [source, setSource] = useState<SourceFilter>('all');
  const [sortField, setSortField] = useState<SortField>('title');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
  const [page, setPage] = useState(1);
  const pageSize = 20;

  // Related papers modal state
  const [relatedModalOpen, setRelatedModalOpen] = useState(false);
  const [selectedPaperForRelated, setSelectedPaperForRelated] = useState<{ id: string; title: string } | null>(null);
  const [relatedPapers, setRelatedPapers] = useState<RelatedPaper[]>([]);
  const [relatedLoading, setRelatedLoading] = useState(false);
  const [relatedError, setRelatedError] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1); // Reset to first page on search
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  // Fetch papers with filters
  const { data: papersData, isLoading, error } = useQuery({
    queryKey: ['knowledge-base-papers', debouncedSearch, category, source, sortField, sortOrder, page],
    queryFn: () => apiClient.listPapers({
      search: debouncedSearch || undefined,
      category: category || undefined,
      source: source === 'all' ? undefined : source,
      sort: sortField,
      sort_order: sortOrder,
      page,
      page_size: pageSize
    })
  });

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['knowledge-base-stats'],
    queryFn: () => apiClient.getKnowledgeBaseStats()
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (paperId: string) => apiClient.deletePaper(paperId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] });
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-stats'] });
    }
  });

  // Clear mutation
  const clearMutation = useMutation({
    mutationFn: () => apiClient.clearKnowledgeBase(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] });
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-stats'] });
    }
  });

  const handleDelete = (paperId: string, title: string) => {
    if (confirm(`Delete "${title}"?`)) {
      deleteMutation.mutate(paperId);
    }
  };

  const handleViewPdf = (paperId: string) => {
    const pdfUrl = apiClient.getPaperPdfUrl(paperId);
    window.open(pdfUrl, '_blank', 'noopener,noreferrer');
  };

  const handleClearAll = () => {
    if (confirm('This will delete ALL papers and vectors. Are you sure?')) {
      if (confirm('This action cannot be undone. Proceed?')) {
        clearMutation.mutate();
      }
    }
  };

  const handleFindRelatedPapers = async (paperId: string, paperTitle: string) => {
    setSelectedPaperForRelated({ id: paperId, title: paperTitle });
    setRelatedModalOpen(true);
    setRelatedLoading(true);
    setRelatedError(null);
    setRelatedPapers([]);

    try {
      const response = await apiClient.getRelatedPapers(paperId);
      setRelatedPapers(response.related_papers || []);
    } catch (err) {
      setRelatedError(err instanceof Error ? err.message : 'Failed to fetch related papers');
    } finally {
      setRelatedLoading(false);
    }
  };

  const handleRelatedModalClose = () => {
    setRelatedModalOpen(false);
    setSelectedPaperForRelated(null);
    setRelatedPapers([]);
    setRelatedError(null);
  };

  const handleRelatedIndexSuccess = () => {
    // Refresh the papers list after indexing related papers
    queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] });
    queryClient.invalidateQueries({ queryKey: ['knowledge-base-stats'] });
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
    setPage(1);
  };

  const resetFilters = useCallback(() => {
    setSearch('');
    setDebouncedSearch('');
    setCategory('');
    setSource('all');
    setSortField('title');
    setSortOrder('asc');
    setPage(1);
  }, []);

  const papers = papersData?.papers || [];
  const totalPages = papersData?.total_pages || 1;
  const availableCategories = papersData?.available_categories || [];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Knowledge Base</h1>
          <p className="text-muted-foreground">
            Manage your indexed research papers
          </p>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Total Papers</p>
              <p className="text-2xl font-bold">{stats.total_papers}</p>
            </div>
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Total Vectors</p>
              <p className="text-2xl font-bold">{stats.total_vectors.toLocaleString()}</p>
            </div>
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Avg Chunks/Paper</p>
              <p className="text-2xl font-bold">{stats.avg_chunks_per_paper.toFixed(1)}</p>
            </div>
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Categories</p>
              <p className="text-2xl font-bold">{Object.keys(stats.categories).length}</p>
            </div>
          </div>
        )}

        {/* Filters Row */}
        <div className="mb-6 p-4 border border-border rounded-lg bg-card">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* Search */}
            <div className="lg:col-span-2">
              <label className="block text-sm font-medium text-muted-foreground mb-1">Search</label>
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by title..."
                className="w-full px-4 py-2 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            {/* Category Filter */}
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1">Category</label>
              <select
                value={category}
                onChange={(e) => { setCategory(e.target.value); setPage(1); }}
                className="w-full px-4 py-2 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="">All Categories</option>
                {availableCategories.slice(0, 20).map((cat) => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            {/* Source Filter */}
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1">Source</label>
              <select
                value={source}
                onChange={(e) => { setSource(e.target.value as SourceFilter); setPage(1); }}
                className="w-full px-4 py-2 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="all">All Papers</option>
                <option value="kb">Knowledge Base</option>
                <option value="linked">Linked Papers</option>
              </select>
            </div>

            {/* Sort */}
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1">Sort By</label>
              <select
                value={`${sortField}-${sortOrder}`}
                onChange={(e) => {
                  const [field, order] = e.target.value.split('-') as [SortField, SortOrder];
                  setSortField(field);
                  setSortOrder(order);
                  setPage(1);
                }}
                className="w-full px-4 py-2 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="title-asc">Title (A-Z)</option>
                <option value="title-desc">Title (Z-A)</option>
                <option value="date-desc">Published: Newest</option>
                <option value="date-asc">Published: Oldest</option>
                <option value="date_added-desc">Added: Recent</option>
                <option value="date_added-asc">Added: Oldest</option>
                <option value="chunks-desc">Most Chunks</option>
                <option value="chunks-asc">Fewest Chunks</option>
              </select>
            </div>
          </div>

          {/* Filter Actions */}
          <div className="mt-4 flex flex-wrap gap-2 items-center justify-between">
            <div className="flex gap-2">
              <button
                onClick={resetFilters}
                className="px-3 py-1.5 text-sm border border-border rounded-lg hover:bg-muted transition-colors"
              >
                Reset Filters
              </button>
              {(debouncedSearch || category || source !== 'all') && (
                <span className="text-sm text-muted-foreground self-center">
                  Filtered: {papersData?.total_papers || 0} papers
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] })}
                className="px-4 py-2 border border-border rounded-lg hover:bg-muted transition-colors"
              >
                Refresh
              </button>
              <button
                onClick={handleClearAll}
                disabled={clearMutation.isPending || !papers.length}
                className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Clear All Papers
              </button>
            </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive rounded-lg mb-6">
            <p className="text-destructive font-medium">Error loading papers</p>
            <p className="text-sm text-destructive/80">{(error as any)?.message}</p>
          </div>
        )}

        {/* Table */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-border border-t-primary"></div>
            <p className="mt-4 text-muted-foreground">Loading papers...</p>
          </div>
        ) : papers.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg">
              {debouncedSearch || category || source !== 'all'
                ? 'No papers match your filters'
                : 'No papers in knowledge base'}
            </p>
            <p className="text-sm mt-2">
              {debouncedSearch || category || source !== 'all'
                ? 'Try adjusting your filters'
                : 'Search and index papers from the Discover page'}
            </p>
          </div>
        ) : (
          <div className="border border-border rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted">
                  <tr>
                    <th
                      className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:bg-muted/80"
                      onClick={() => handleSort('title')}
                    >
                      <div className="flex items-center gap-2">
                        Title
                        {sortField === 'title' && (sortOrder === 'asc' ? '↑' : '↓')}
                      </div>
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                      Authors
                    </th>
                    <th
                      className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:bg-muted/80"
                      onClick={() => handleSort('chunks')}
                    >
                      <div className="flex items-center gap-2">
                        Chunks
                        {sortField === 'chunks' && (sortOrder === 'asc' ? '↑' : '↓')}
                      </div>
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                      Category
                    </th>
                    <th
                      className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:bg-muted/80"
                      onClick={() => handleSort('date')}
                    >
                      <div className="flex items-center gap-2">
                        Published
                        {sortField === 'date' && (sortOrder === 'asc' ? '↑' : '↓')}
                      </div>
                    </th>
                    <th
                      className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:bg-muted/80"
                      onClick={() => handleSort('date_added')}
                    >
                      <div className="flex items-center gap-2">
                        Added
                        {sortField === 'date_added' && (sortOrder === 'asc' ? '↑' : '↓')}
                      </div>
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {papers.map((paper: Paper) => {
                    const paperId = paper.paper_id || paper.arxiv_id || '';
                    const isLinked = (paper as any).is_linked;
                    return (
                      <tr key={paperId} className={`hover:bg-muted/50 ${isLinked ? 'bg-blue-50/30' : ''}`}>
                        <td className="px-4 py-3">
                          <div className="max-w-md">
                            <p className="font-medium text-sm line-clamp-2">{paper.title}</p>
                            {isLinked && (
                              <span className="text-xs text-blue-600 bg-blue-100 px-1.5 py-0.5 rounded mt-1 inline-block">
                                Linked
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-muted-foreground">
                            {paper.authors.slice(0, 2).join(', ')}
                            {paper.authors.length > 2 && ` +${paper.authors.length - 2}`}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-center">{paper.chunk_count || 'N/A'}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs text-muted-foreground">
                            {paper.categories && paper.categories[0] ? paper.categories[0] : 'N/A'}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs text-muted-foreground">
                            {paper.published_date || 'N/A'}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs text-muted-foreground">
                            {(paper as any).indexed_at
                              ? new Date((paper as any).indexed_at).toLocaleDateString()
                              : 'N/A'}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleViewPdf(paperId)}
                              disabled={!paperId}
                              className="text-xs px-3 py-1.5 bg-primary text-primary-foreground hover:bg-primary/90 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title="View PDF"
                            >
                              View PDF
                            </button>
                            <button
                              onClick={() => handleFindRelatedPapers(paperId, paper.title)}
                              disabled={!paperId}
                              className="text-xs px-2 py-1.5 border border-border hover:bg-muted rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title="Find papers cited by this paper"
                            >
                              Related
                            </button>
                            <button
                              onClick={() => handleDelete(paperId, paper.title)}
                              disabled={deleteMutation.isPending || !paperId}
                              className="text-xs px-2 py-1 text-destructive hover:bg-destructive/10 rounded transition-colors disabled:opacity-50"
                            >
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-6 flex items-center justify-center gap-4">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-4 py-2 border border-border rounded-lg hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <span className="text-sm text-muted-foreground">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-4 py-2 border border-border rounded-lg hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        )}

        {/* Results Count */}
        {papersData && papersData.total_papers > 0 && (
          <div className="mt-4 text-sm text-muted-foreground text-center">
            Showing {papers.length} of {papersData.total_papers} papers
          </div>
        )}
      </div>

      {/* Related Papers Modal */}
      <RelatedPapersModal
        isOpen={relatedModalOpen}
        onClose={handleRelatedModalClose}
        paperId={selectedPaperForRelated?.id || ''}
        paperTitle={selectedPaperForRelated?.title || ''}
        relatedPapers={relatedPapers}
        isLoading={relatedLoading}
        error={relatedError}
        onIndexSuccess={handleRelatedIndexSuccess}
      />
    </div>
  );
}
