/**
 * RelatedPapersModal Component
 *
 * Modal for viewing and indexing related/cited papers from a KB paper.
 * Users can select which papers to index into their knowledge base.
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../services/api/client';
import type { Paper } from '../types';

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

interface RelatedPapersModalProps {
  isOpen: boolean;
  onClose: () => void;
  paperId: string;
  paperTitle: string;
  relatedPapers: RelatedPaper[];
  isLoading: boolean;
  error: string | null;
  onIndexSuccess?: () => void;
}

export default function RelatedPapersModal({
  isOpen,
  onClose,
  paperId: _paperId,  // Keep for future use (e.g., logging)
  paperTitle,
  relatedPapers,
  isLoading,
  error,
  onIndexSuccess
}: RelatedPapersModalProps) {
  // Suppress unused variable warning - paperId kept for potential future features
  void _paperId;
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());

  // Index mutation
  const indexMutation = useMutation({
    mutationFn: (papers: Paper[]) => apiClient.indexPapers(papers),
    onSuccess: () => {
      setSelectedPapers(new Set());
      onIndexSuccess?.();
    }
  });

  const toggleSelection = (arxivId: string) => {
    const newSelection = new Set(selectedPapers);
    if (newSelection.has(arxivId)) {
      newSelection.delete(arxivId);
    } else {
      newSelection.add(arxivId);
    }
    setSelectedPapers(newSelection);
  };

  const selectAllNew = () => {
    const newPapers = relatedPapers
      .filter(p => !p.already_indexed)
      .map(p => p.arxiv_id);
    setSelectedPapers(new Set(newPapers));
  };

  const handleIndex = () => {
    const papersToIndex = relatedPapers
      .filter(p => selectedPapers.has(p.arxiv_id))
      .map(p => ({
        arxiv_id: p.arxiv_id,
        title: p.title,
        authors: p.authors,
        abstract: p.abstract,
        published_date: p.published_date,
        categories: p.categories,
        url: p.url
      }));

    if (papersToIndex.length > 0) {
      indexMutation.mutate(papersToIndex as Paper[]);
    }
  };

  if (!isOpen) return null;

  const indexablePapers = relatedPapers.filter(p => !p.already_indexed);
  const selectedCount = selectedPapers.size;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-background border border-border rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] flex flex-col m-4">
        {/* Header */}
        <div className="p-4 border-b border-border flex-shrink-0">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-lg font-semibold">Related Papers</h2>
              <p className="text-sm text-muted-foreground mt-1 line-clamp-1">
                Citations from: {paperTitle}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground text-xl leading-none"
            >
              &times;
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-4 border-border border-t-primary"></div>
              <span className="ml-3 text-muted-foreground">
                Extracting citations and searching ArXiv...
              </span>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-destructive font-medium">Failed to load related papers</p>
              <p className="text-sm text-muted-foreground mt-1">{error}</p>
            </div>
          ) : relatedPapers.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No related papers found</p>
              <p className="text-sm text-muted-foreground mt-1">
                The paper may not have extractable citations, or cited papers aren't on ArXiv.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Select All Button */}
              {indexablePapers.length > 0 && (
                <div className="flex items-center justify-between mb-4">
                  <span className="text-sm text-muted-foreground">
                    {indexablePapers.length} paper{indexablePapers.length > 1 ? 's' : ''} available to index
                  </span>
                  <button
                    onClick={selectAllNew}
                    className="text-sm text-primary hover:underline"
                  >
                    Select all new
                  </button>
                </div>
              )}

              {/* Paper List */}
              {relatedPapers.map((paper) => (
                <div
                  key={paper.arxiv_id}
                  className={`p-4 border rounded-lg transition-colors ${
                    paper.already_indexed
                      ? 'border-green-200 bg-green-50/50'
                      : selectedPapers.has(paper.arxiv_id)
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-primary/50'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    {/* Checkbox */}
                    <input
                      type="checkbox"
                      checked={paper.already_indexed || selectedPapers.has(paper.arxiv_id)}
                      onChange={() => !paper.already_indexed && toggleSelection(paper.arxiv_id)}
                      disabled={paper.already_indexed}
                      className="mt-1 h-4 w-4 rounded border-border disabled:opacity-50"
                    />

                    <div className="flex-1 min-w-0">
                      {/* Title */}
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="font-medium text-sm line-clamp-2">{paper.title}</h3>
                        {paper.already_indexed && (
                          <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded shrink-0">
                            In KB
                          </span>
                        )}
                      </div>

                      {/* Authors */}
                      <p className="text-xs text-muted-foreground mt-1">
                        {paper.authors.slice(0, 3).join(', ')}
                        {paper.authors.length > 3 && ` +${paper.authors.length - 3}`}
                        {paper.published_date && ` (${paper.published_date.slice(0, 4)})`}
                      </p>

                      {/* Relevance */}
                      {paper.relevance && (
                        <p className="text-xs text-primary mt-2 italic">
                          {paper.relevance}
                        </p>
                      )}

                      {/* Abstract Preview */}
                      <p className="text-xs text-muted-foreground mt-2 line-clamp-2">
                        {paper.abstract}
                      </p>

                      {/* Categories + Link */}
                      <div className="flex items-center justify-between mt-2">
                        <div className="flex gap-1">
                          {paper.categories.slice(0, 2).map((cat) => (
                            <span
                              key={cat}
                              className="text-xs bg-muted px-1.5 py-0.5 rounded"
                            >
                              {cat}
                            </span>
                          ))}
                        </div>
                        <a
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-primary hover:underline"
                        >
                          View on ArXiv &rarr;
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border flex-shrink-0">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">
              {selectedCount > 0
                ? `${selectedCount} paper${selectedCount > 1 ? 's' : ''} selected`
                : 'Select papers to index'}
            </span>
            <div className="flex gap-2">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm border border-border rounded-lg hover:bg-muted transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleIndex}
                disabled={selectedCount === 0 || indexMutation.isPending}
                className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {indexMutation.isPending
                  ? 'Indexing...'
                  : `Index ${selectedCount || ''} Selected`}
              </button>
            </div>
          </div>

          {/* Success Message */}
          {indexMutation.isSuccess && (
            <p className="text-sm text-green-600 mt-2">
              Papers indexed successfully! They will appear in your knowledge base shortly.
            </p>
          )}

          {/* Error Message */}
          {indexMutation.isError && (
            <p className="text-sm text-destructive mt-2">
              Failed to index papers: {(indexMutation.error as any)?.message || 'Unknown error'}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
