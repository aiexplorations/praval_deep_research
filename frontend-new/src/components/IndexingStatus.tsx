/**
 * IndexingStatus Component
 *
 * Displays real-time paper indexing progress, success notifications,
 * and error messages from the backend via SSE.
 */

import { useIndexingStatus } from '../hooks/useIndexingStatus';

interface IndexingStatusProps {
  onComplete?: () => void;
}

export function IndexingStatus({ onComplete }: IndexingStatusProps) {
  const { progress, isConnected, resetProgress } = useIndexingStatus();

  // Call onComplete when indexing finishes
  if (progress.stage === 'complete' && onComplete) {
    // Use setTimeout to avoid calling during render
    setTimeout(onComplete, 100);
  }

  // Don't show anything if idle
  if (progress.stage === 'idle') {
    return null;
  }

  const isProcessing = progress.stage === 'starting' || progress.stage === 'processing';
  const isComplete = progress.stage === 'complete';
  const hasErrors = progress.failed > 0 || progress.errors.length > 0;

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-md">
      {/* Main Status Card */}
      <div className={`rounded-lg shadow-lg border p-4 ${
        hasErrors && isComplete
          ? 'bg-yellow-50 border-yellow-300'
          : isComplete
            ? 'bg-green-50 border-green-300'
            : 'bg-white border-border'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {isProcessing && (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent" />
            )}
            {isComplete && !hasErrors && (
              <span className="text-green-600 text-lg">✓</span>
            )}
            {isComplete && hasErrors && (
              <span className="text-yellow-600 text-lg">⚠</span>
            )}
            <span className="font-medium text-sm">
              {isProcessing ? 'Indexing Papers...' : 'Indexing Complete'}
            </span>
          </div>

          {/* Connection indicator */}
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            {isComplete && (
              <button
                onClick={resetProgress}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Dismiss
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar (when processing) */}
        {isProcessing && progress.total > 0 && (
          <div className="mb-2">
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${(progress.current / progress.total) * 100}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {progress.current} / {progress.total} papers
            </p>
          </div>
        )}

        {/* Current Paper */}
        {isProcessing && progress.currentPaper && (
          <p className="text-xs text-muted-foreground truncate mb-2">
            Processing: {progress.currentPaper}
          </p>
        )}

        {/* Stats */}
        <div className="flex gap-4 text-xs">
          <div>
            <span className="text-green-600 font-medium">{progress.papersIndexed}</span>
            <span className="text-muted-foreground ml-1">indexed</span>
          </div>
          <div>
            <span className="text-blue-600 font-medium">{progress.vectorsStored}</span>
            <span className="text-muted-foreground ml-1">vectors</span>
          </div>
          {progress.failed > 0 && (
            <div>
              <span className="text-red-600 font-medium">{progress.failed}</span>
              <span className="text-muted-foreground ml-1">failed</span>
            </div>
          )}
        </div>

        {/* Error List */}
        {hasErrors && progress.errors.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border">
            <p className="text-xs font-medium text-red-600 mb-1">Errors:</p>
            <div className="max-h-32 overflow-y-auto">
              {progress.errors.map((error, idx) => (
                <p key={idx} className="text-xs text-red-500 mb-1 line-clamp-2">
                  • {error}
                </p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default IndexingStatus;
