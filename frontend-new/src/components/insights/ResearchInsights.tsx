/**
 * Research Insights Component
 *
 * Displays proactive research insights including:
 * - Research area clusters
 * - Trending topics
 * - Research gaps
 * - Personalized next steps
 * - Suggested papers from arXiv
 */

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api/client';

interface ResearchInsightsProps {
  onRefresh?: () => void;
}

export default function ResearchInsights({ onRefresh }: ResearchInsightsProps) {
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
        <h3 className="text-lg font-semibold mb-2">🔍 Research Insights</h3>
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
        <h3 className="text-lg font-semibold">🔍 Research Insights</h3>
        <button
          onClick={handleRefresh}
          className="text-xs px-3 py-1 border border-border rounded hover:bg-muted transition-colors"
          title="Refresh insights"
        >
          🔄 Refresh
        </button>
      </div>

      {/* Research Areas */}
      {data.research_areas.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">📚 Your Research Areas</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.research_areas.map((area, idx) => (
              <div key={idx} className="p-3 bg-muted/50 rounded border border-border">
                <div className="flex items-start justify-between">
                  <h5 className="font-medium text-sm">{area.name}</h5>
                  <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                    {area.paper_count} papers
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">{area.description}</p>
                {area.significance && (
                  <p className="text-xs text-primary mt-2">💡 {area.significance}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trending Topics */}
      {data.trending_topics.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">🔥 Trending Topics</h4>
          <div className="flex flex-wrap gap-2">
            {data.trending_topics.map((topic, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-primary/10 text-primary text-xs rounded-full border border-primary/20"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Research Gaps */}
      {data.research_gaps.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">💡 Research Opportunities</h4>
          <div className="space-y-2">
            {data.research_gaps.map((gap, idx) => (
              <div key={idx} className="p-3 bg-muted/30 rounded border border-dashed border-border">
                <h5 className="font-medium text-sm text-primary">{gap.gap_title}</h5>
                <p className="text-xs text-muted-foreground mt-1">{gap.description}</p>
                {gap.potential_value && (
                  <p className="text-xs text-muted-foreground mt-1">
                    <span className="font-medium">Value:</span> {gap.potential_value}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next Steps */}
      {data.next_steps.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-3">✅ Suggested Next Steps</h4>
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
                        ⏱️ {step.estimated_time}
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
    </div>
  );
}
