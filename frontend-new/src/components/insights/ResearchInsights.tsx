/**
 * Research Insights Component
 *
 * Displays proactive research insights including:
 * - Research area clusters (expandable with papers)
 * - Trending topics
 * - Research gaps
 * - Personalized next steps
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../../services/api/client';
import { useChatStore } from '../../store/useChatStore';
import RelatedPapersModal from '../RelatedPapersModal';

interface ResearchInsightsProps {
  onRefresh?: () => void;
  onTopicClick?: (topic: string) => void;
}

interface AreaPaper {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  relevance_score: number;
}

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

export default function ResearchInsights({ onRefresh, onTopicClick }: ResearchInsightsProps) {
  const [expandedArea, setExpandedArea] = useState<number | null>(null);
  const [summarizingPaper, setSummarizingPaper] = useState<string | null>(null);

  // Related papers modal state
  const [relatedModalOpen, setRelatedModalOpen] = useState(false);
  const [relatedModalPaperId, setRelatedModalPaperId] = useState<string>('');
  const [relatedModalPaperTitle, setRelatedModalPaperTitle] = useState<string>('');
  const [relatedPapers, setRelatedPapers] = useState<RelatedPaper[]>([]);
  const [relatedLoading, setRelatedLoading] = useState(false);
  const [relatedError, setRelatedError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { setCurrentConversation, clearMessages, setPendingQuestion } = useChatStore();
  const [areaPapers, setAreaPapers] = useState<Record<string, AreaPaper[]>>({});
  const [loadingArea, setLoadingArea] = useState<string | null>(null);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['research-insights'],
    queryFn: () => apiClient.getResearchInsights(),
    staleTime: 30 * 60 * 1000, // 30 minutes
    retry: 1
  });

  const handleRefresh = () => {
    refetch();
    onRefresh?.();
  };

  const handleFindRelatedPapers = async (paperId: string, paperTitle: string) => {
    setRelatedModalPaperId(paperId);
    setRelatedModalPaperTitle(paperTitle);
    setRelatedModalOpen(true);
    setRelatedLoading(true);
    setRelatedError(null);
    setRelatedPapers([]);

    try {
      const response = await apiClient.getRelatedPapers(paperId);
      setRelatedPapers(response.related_papers);
    } catch (err: any) {
      setRelatedError(err?.message || 'Failed to load related papers');
    } finally {
      setRelatedLoading(false);
    }
  };

  const toggleAreaExpand = async (idx: number, areaName: string) => {
    if (expandedArea === idx) {
      setExpandedArea(null);
      return;
    }

    setExpandedArea(idx);

    // Load papers if not cached
    if (!areaPapers[areaName]) {
      setLoadingArea(areaName);
      try {
        const response = await apiClient.getAreaPapers(areaName, 10);
        setAreaPapers(prev => ({
          ...prev,
          [areaName]: response.papers
        }));
      } catch (err) {
        console.error('Failed to load area papers:', err);
      } finally {
        setLoadingArea(null);
      }
    }
  };

  if (isLoading) {
    return (
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-4 border-border border-t-primary"></div>
          <span className="ml-3 text-muted-foreground">Analyzing your research activity...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-destructive/10 border border-destructive rounded-lg p-6">
        <p className="text-destructive font-medium">Failed to load insights</p>
        <p className="text-sm text-destructive/80 mt-1">{(error as any)?.message}</p>
        <button
          onClick={handleRefresh}
          className="mt-3 px-4 py-2 bg-destructive text-destructive-foreground rounded hover:bg-destructive/90 text-sm"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!data) return null;

  const hasInsights = data.research_areas.length > 0 ||
    data.trending_topics.length > 0 ||
    data.research_gaps.length > 0 ||
    data.next_steps.length > 0;

  if (!hasInsights && data.kb_context.total_papers === 0) {
    return (
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-2">Research Insights</h3>
        <p className="text-muted-foreground text-sm">
          Index some papers to see personalized research insights and recommendations.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card border border-border rounded-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Research Insights</h3>
        <button
          onClick={handleRefresh}
          className="text-xs px-3 py-1 border border-border rounded hover:bg-muted transition-colors"
          title="Refresh insights"
        >
          Refresh
        </button>
      </div>

      {/* Research Areas - Expandable Cards */}
      {data.research_areas.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">Your Research Areas</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.research_areas.map((area, idx) => {
              const isExpanded = expandedArea === idx;
              const papers = areaPapers[area.name] || [];
              const isLoadingPapers = loadingArea === area.name;

              return (
                <div
                  key={idx}
                  className={`rounded border transition-all ${
                    isExpanded
                      ? 'border-primary bg-primary/5 col-span-1 md:col-span-2'
                      : 'border-border bg-muted/50 hover:border-primary/50 cursor-pointer'
                  }`}
                >
                  {/* Card Header - Always Visible */}
                  <div
                    className={`p-3 ${!isExpanded ? 'cursor-pointer' : ''}`}
                    onClick={() => toggleAreaExpand(idx, area.name)}
                  >
                    <div className="flex items-start justify-between">
                      <h5 className="font-medium text-sm flex items-center gap-2">
                        {area.name}
                        <span className="text-xs text-muted-foreground">
                          {isExpanded ? '▼' : '▶'}
                        </span>
                      </h5>
                      <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                        {area.paper_count} papers
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">{area.description}</p>
                    {area.significance && (
                      <p className="text-xs text-primary mt-2">{area.significance}</p>
                    )}
                    {!isExpanded && (
                      <p className="text-xs text-muted-foreground mt-2 italic">
                        Click to view papers
                      </p>
                    )}
                  </div>

                  {/* Expanded Content - Papers List */}
                  {isExpanded && (
                    <div className="border-t border-border p-3">
                      {isLoadingPapers ? (
                        <div className="flex items-center justify-center py-4">
                          <div className="animate-spin rounded-full h-5 w-5 border-2 border-border border-t-primary"></div>
                          <span className="ml-2 text-xs text-muted-foreground">Loading papers...</span>
                        </div>
                      ) : papers.length === 0 ? (
                        <p className="text-xs text-muted-foreground text-center py-4">
                          No papers found for this area
                        </p>
                      ) : (
                        <div className="space-y-2">
                          <p className="text-xs font-medium text-muted-foreground mb-2">
                            Related Papers ({papers.length})
                          </p>
                          {papers.map((paper) => (
                            <div
                              key={paper.paper_id}
                              className="p-2 bg-background rounded border border-border hover:border-primary/30 transition-colors"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium line-clamp-2">{paper.title}</p>
                                  <p className="text-xs text-muted-foreground mt-1">
                                    {paper.authors.slice(0, 2).join(', ')}
                                    {paper.authors.length > 2 && ` +${paper.authors.length - 2}`}
                                  </p>
                                </div>
                                <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded shrink-0">
                                  {(paper.relevance_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div className="flex items-center gap-2 mt-2 flex-wrap">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    // Open PDF in new tab
                                    const pdfUrl = window.location.hostname === 'localhost'
                                      ? `http://localhost:8000/research/knowledge-base/papers/${paper.paper_id}/pdf`
                                      : `/api/research/knowledge-base/papers/${paper.paper_id}/pdf`;
                                    window.open(pdfUrl, '_blank', 'noopener,noreferrer');
                                  }}
                                  className="text-xs px-2 py-1 bg-primary text-primary-foreground hover:bg-primary/90 rounded transition-colors"
                                >
                                  Read PDF
                                </button>
                                <button
                                  onClick={async (e) => {
                                    e.stopPropagation();
                                    setSummarizingPaper(paper.paper_id);
                                    try {
                                      // Create conversation title
                                      const conversationTitle = `Summary of paper: ${paper.title.slice(0, 60)}${paper.title.length > 60 ? '...' : ''}`;

                                      // Find or create conversation with this title
                                      const { id, created } = await apiClient.findOrCreateConversation(conversationTitle);

                                      // Set up chat state
                                      setCurrentConversation(id);

                                      if (created) {
                                        // New conversation - clear messages and set pending question
                                        clearMessages();
                                        const summaryQuestion = `Please provide a comprehensive summary of the paper "${paper.title}". Include the main contributions, methodology, key findings, and conclusions.`;
                                        setPendingQuestion(summaryQuestion);
                                      }

                                      // Navigate to chat
                                      navigate('/chat');
                                    } catch (err) {
                                      console.error('Failed to create summary chat:', err);
                                    } finally {
                                      setSummarizingPaper(null);
                                    }
                                  }}
                                  disabled={summarizingPaper === paper.paper_id}
                                  className="text-xs px-2 py-1 bg-secondary text-secondary-foreground hover:bg-secondary/80 rounded transition-colors disabled:opacity-50"
                                >
                                  {summarizingPaper === paper.paper_id ? 'Opening...' : 'Summarize in Chat'}
                                </button>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onTopicClick?.(paper.title);
                                  }}
                                  className="text-xs text-primary hover:underline"
                                >
                                  Ask about this paper &rarr;
                                </button>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleFindRelatedPapers(paper.paper_id, paper.title);
                                  }}
                                  className="text-xs px-2 py-1 border border-primary/30 text-primary hover:bg-primary/10 rounded transition-colors"
                                >
                                  Find Related Papers
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      <button
                        onClick={() => setExpandedArea(null)}
                        className="w-full mt-3 text-xs text-muted-foreground hover:text-foreground text-center py-1"
                      >
                        &#9650; Collapse
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Trending Topics */}
      {data.trending_topics.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">Trending Topics</h4>
          <div className="flex flex-wrap gap-2">
            {data.trending_topics.map((topic, idx) => (
              <button
                key={idx}
                onClick={() => onTopicClick?.(topic)}
                className="px-3 py-1 bg-primary/10 text-primary text-xs rounded-full border border-primary/20 hover:bg-primary/20 transition-colors cursor-pointer"
                title={`Search for: ${topic}`}
              >
                {topic}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Research Gaps */}
      {data.research_gaps.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">Research Opportunities</h4>
          <div className="space-y-2">
            {data.research_gaps.map((gap, idx) => (
              <button
                key={idx}
                onClick={() => onTopicClick?.(gap.gap_title)}
                className="w-full text-left p-3 bg-muted/30 rounded border border-dashed border-border hover:bg-muted/50 hover:border-primary/30 transition-all cursor-pointer"
                title={`Search for: ${gap.gap_title}`}
              >
                <h5 className="font-medium text-sm text-primary flex items-center gap-2">
                  {gap.gap_title}
                  <span className="text-xs opacity-50">→ Click to search</span>
                </h5>
                <p className="text-xs text-muted-foreground mt-1">{gap.description}</p>
                {gap.potential_value && (
                  <p className="text-xs text-muted-foreground mt-1">
                    <span className="font-medium">Value:</span> {gap.potential_value}
                  </p>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Next Steps */}
      {data.next_steps.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">Suggested Next Steps</h4>
          <div className="space-y-2">
            {data.next_steps.map((step, idx) => (
              <div key={idx} className="p-3 bg-primary/5 rounded border border-primary/20">
                <div className="flex items-start gap-2">
                  <span className="text-primary font-bold text-sm">{idx + 1}.</span>
                  <div className="flex-1">
                    <p className="text-sm font-medium">{step.action}</p>
                    <p className="text-xs text-muted-foreground mt-1">{step.rationale}</p>
                    {step.estimated_time && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {step.estimated_time}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="text-xs text-muted-foreground pt-4 border-t border-border">
        <p>
          Analyzed {data.kb_context.total_papers} papers •
          Quality: {data.generation_metadata.insights_quality} •
          Updated: {new Date(data.generation_metadata.timestamp).toLocaleString()}
        </p>
      </div>

      {/* Related Papers Modal */}
      <RelatedPapersModal
        isOpen={relatedModalOpen}
        onClose={() => setRelatedModalOpen(false)}
        paperId={relatedModalPaperId}
        paperTitle={relatedModalPaperTitle}
        relatedPapers={relatedPapers}
        isLoading={relatedLoading}
        error={relatedError}
        onIndexSuccess={() => {
          // Refresh insights after indexing related papers
          refetch();
        }}
      />
    </div>
  );
}
