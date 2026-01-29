import { useState, useCallback, useRef, useEffect } from 'react';
import { analyzeWithProgress, type AnalysisProgress, type SSEConnection } from '../services/sse';

interface StreamingAnalysisState {
  isStreaming: boolean;
  progress: number;
  currentStage: string | null;
  taskId: string | null;
  error: string | null;
}

interface UseStreamingAnalysisReturn extends StreamingAnalysisState {
  startAnalysis: (symbol: string) => void;
  cancelAnalysis: () => void;
}

const STAGE_LABELS: Record<string, string> = {
  data_collection: 'Collecting market data',
  initial_analysis: 'Performing initial analysis',
  deep_analysis: 'Running deep analysis',
  risk_assessment: 'Assessing risks',
  strategy_synthesis: 'Synthesizing strategy',
};

export function useStreamingAnalysis(
  onComplete?: (result: AnalysisProgress) => void,
  onError?: (error: Error) => void
): UseStreamingAnalysisReturn {
  const [state, setState] = useState<StreamingAnalysisState>({
    isStreaming: false,
    progress: 0,
    currentStage: null,
    taskId: null,
    error: null,
  });

  const connectionRef = useRef<SSEConnection | null>(null);

  const startAnalysis = useCallback(
    async (symbol: string) => {
      // Close any existing connection
      if (connectionRef.current) {
        connectionRef.current.close();
      }

      setState({
        isStreaming: true,
        progress: 0,
        currentStage: 'Starting analysis',
        taskId: null,
        error: null,
      });

      connectionRef.current = await analyzeWithProgress(symbol, {
        onProgress: (data) => {
          setState((prev) => ({
            ...prev,
            progress: data.progress * 100,
            currentStage: data.current_stage
              ? STAGE_LABELS[data.current_stage] || data.current_stage
              : prev.currentStage,
            taskId: data.task_id,
          }));
        },
        onComplete: (result) => {
          setState((prev) => ({
            ...prev,
            isStreaming: false,
            progress: 100,
            currentStage: result.status === 'completed' ? 'Analysis complete' : 'Analysis ended',
          }));
          onComplete?.(result);
        },
        onError: (error) => {
          setState((prev) => ({
            ...prev,
            isStreaming: false,
            error: error.message,
          }));
          onError?.(error);
        },
      });
    },
    [onComplete, onError]
  );

  const cancelAnalysis = useCallback(() => {
    if (connectionRef.current) {
      connectionRef.current.close();
      connectionRef.current = null;
    }
    setState((prev) => ({
      ...prev,
      isStreaming: false,
      currentStage: 'Analysis cancelled',
    }));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (connectionRef.current) {
        connectionRef.current.close();
      }
    };
  }, []);

  return {
    ...state,
    startAnalysis,
    cancelAnalysis,
  };
}
