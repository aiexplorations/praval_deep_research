/**
 * Hook for tracking paper indexing status via SSE
 *
 * Connects to the backend SSE endpoint to receive real-time
 * indexing progress, success, and error notifications.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface IndexingProgress {
  stage: 'idle' | 'starting' | 'processing' | 'complete' | 'error';
  current: number;
  total: number;
  currentPaper: string | null;
  papersIndexed: number;
  vectorsStored: number;
  failed: number;
  errors: string[];
}

interface SSEEvent {
  event_type: string;
  [key: string]: any;
}

const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : '';

export function useIndexingStatus() {
  const [progress, setProgress] = useState<IndexingProgress>({
    stage: 'idle',
    current: 0,
    total: 0,
    currentPaper: null,
    papersIndexed: 0,
    vectorsStored: 0,
    failed: 0,
    errors: []
  });

  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    // Don't reconnect if already connected
    if (eventSourceRef.current?.readyState === EventSource.OPEN) {
      return;
    }

    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource(`${API_BASE_URL}/sse/agent-updates`);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      console.log('SSE connected');
    };

    eventSource.onmessage = (event) => {
      try {
        const data: SSEEvent = JSON.parse(event.data);
        handleEvent(data);
      } catch (e) {
        console.error('Failed to parse SSE event:', e);
      }
    };

    eventSource.onerror = () => {
      setIsConnected(false);
      eventSource.close();

      // Reconnect after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('SSE reconnecting...');
        connect();
      }, 5000);
    };
  }, []);

  const handleEvent = useCallback((data: SSEEvent) => {
    switch (data.event_type) {
      case 'indexing_progress':
        setProgress(prev => ({
          ...prev,
          stage: data.stage === 'starting' ? 'starting' : 'processing',
          current: data.current || 0,
          total: data.total || 0,
          currentPaper: data.current_paper || null
        }));
        break;

      case 'paper_indexed':
        setProgress(prev => ({
          ...prev,
          papersIndexed: prev.papersIndexed + 1,
          vectorsStored: prev.vectorsStored + (data.vectors_stored || 0)
        }));
        break;

      case 'indexing_error':
        setProgress(prev => ({
          ...prev,
          failed: prev.failed + 1,
          errors: [...prev.errors, `${data.title}: ${data.error}`].slice(-10) // Keep last 10 errors
        }));
        break;

      case 'indexing_complete':
        setProgress(prev => ({
          ...prev,
          stage: 'complete',
          papersIndexed: data.papers_indexed || 0,
          vectorsStored: data.vectors_stored || 0,
          failed: data.failed || 0,
          errors: data.errors || prev.errors
        }));
        break;

      case 'linked_paper_indexed':
        // Optional: track linked paper indexing separately
        break;

      default:
        // Ignore other event types
        break;
    }
  }, []);

  const resetProgress = useCallback(() => {
    setProgress({
      stage: 'idle',
      current: 0,
      total: 0,
      currentPaper: null,
      papersIndexed: 0,
      vectorsStored: 0,
      failed: 0,
      errors: []
    });
  }, []);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    progress,
    isConnected,
    resetProgress,
    connect,
    disconnect
  };
}

export default useIndexingStatus;
