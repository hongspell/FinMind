/**
 * Server-Sent Events (SSE) utility for real-time analysis progress
 */

export interface SSEEvent<T = unknown> {
  event?: string;
  data: T;
}

export interface AnalysisProgress {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  target: string;
  chain: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  current_stage?: string;
  result?: unknown;
  error?: string;
}

export type ProgressCallback = (progress: AnalysisProgress) => void;
export type ErrorCallback = (error: Error) => void;
export type CompleteCallback = (result: AnalysisProgress) => void;

export interface SSEConnection {
  close: () => void;
}

/**
 * Connect to SSE stream for analysis progress updates
 */
export function subscribeToAnalysisProgress(
  taskId: string,
  callbacks: {
    onProgress?: ProgressCallback;
    onComplete?: CompleteCallback;
    onError?: ErrorCallback;
  }
): SSEConnection {
  const { onProgress, onComplete, onError } = callbacks;

  const eventSource = new EventSource(`/api/v1/analyze/${taskId}/stream`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as AnalysisProgress;

      onProgress?.(data);

      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        onComplete?.(data);
        eventSource.close();
      }
    } catch (e) {
      onError?.(e instanceof Error ? e : new Error('Failed to parse SSE data'));
    }
  };

  eventSource.onerror = () => {
    onError?.(new Error('SSE connection error'));
    eventSource.close();
  };

  // Handle custom error events
  eventSource.addEventListener('error', (event) => {
    if (event instanceof MessageEvent) {
      onError?.(new Error(event.data));
      eventSource.close();
    }
  });

  return {
    close: () => eventSource.close(),
  };
}

/**
 * Create analysis task and subscribe to progress updates
 */
export async function analyzeWithProgress(
  symbol: string,
  callbacks: {
    onProgress?: ProgressCallback;
    onComplete?: CompleteCallback;
    onError?: ErrorCallback;
  }
): Promise<SSEConnection> {
  const { onProgress, onComplete, onError } = callbacks;

  try {
    // Create the analysis task
    const response = await fetch('/api/v1/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        target: symbol.toUpperCase(),
        chain: 'full_analysis',
        parameters: {},
        priority: 5,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create analysis task: ${response.statusText}`);
    }

    const task = await response.json() as AnalysisProgress;

    // Initial progress update
    onProgress?.(task);

    // Subscribe to progress updates
    return subscribeToAnalysisProgress(task.task_id, {
      onProgress,
      onComplete,
      onError,
    });
  } catch (e) {
    const error = e instanceof Error ? e : new Error('Failed to start analysis');
    onError?.(error);
    return { close: () => {} };
  }
}
