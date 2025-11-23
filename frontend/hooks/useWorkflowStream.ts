/**
 * useWorkflowStream - SSE Hook for Real-time Workflow Updates
 * ===========================================================
 * 
 * PRD-28: Replaces WebSocket + Redis + Polling with SSE streaming.
 * Based on successful PRD-27 chat SSE implementation.
 * 
 * Features:
 * - < 500ms latency for updates (vs 2-50 seconds polling)
 * - Automatic reconnection on disconnect
 * - Full subtask details (no truncation)
 * - Clean teardown on unmount
 * - Zero polling required
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export interface SubtaskUpdate {
  subtask_id: number;
  subtask_description: string;  // FULL description (no truncation)
  agent_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'retrying';
  tokens_used?: number;
  execution_time_ms?: number;
  error_message?: string;
  retry_count?: number;
  result_preview?: string;
  result_truncated?: boolean;
  timestamp: string;
}

export interface WorkflowStreamEvent {
  type: 'connected' | 'subtask_update' | 'execution_log' | 'workflow_complete' | 'error';
  data?: any;
  execution_id?: number;
  timestamp: string;
}

interface UseWorkflowStreamOptions {
  executionId: number | null;
  onSubtaskUpdate?: (update: SubtaskUpdate) => void;
  onWorkflowComplete?: (data: any) => void;
  onLog?: (log: any) => void;
  onError?: (error: string) => void;
  autoReconnect?: boolean;
  reconnectDelay?: number;
}

interface UseWorkflowStreamReturn {
  isConnected: boolean;
  error: string | null;
  reconnect: () => void;
  disconnect: () => void;
}

export function useWorkflowStream({
  executionId,
  onSubtaskUpdate,
  onWorkflowComplete,
  onLog,
  onError,
  autoReconnect = true,
  reconnectDelay = 2000,
}: UseWorkflowStreamOptions): UseWorkflowStreamReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const autoDisconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldConnectRef = useRef(true);

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false;
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (autoDisconnectTimeoutRef.current) {
      clearTimeout(autoDisconnectTimeoutRef.current);
      autoDisconnectTimeoutRef.current = null;
    }
    
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const connect = useCallback(() => {
    if (!executionId || !shouldConnectRef.current) {
      return;
    }

    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const url = `${apiUrl}/api/workflows/executions/${executionId}/stream`;
      
      console.log(`🚀 [useWorkflowStream] Connecting to SSE: ${url}`);
      
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onopen = () => {
        console.log('✅ [useWorkflowStream] SSE connected');
        setIsConnected(true);
        setError(null);
      };

      es.onmessage = (event) => {
        try {
          const data: WorkflowStreamEvent = JSON.parse(event.data);
          
          console.log(`📨 [useWorkflowStream] Event received:`, data.type);
          
          switch (data.type) {
            case 'connected':
              console.log('🔗 [useWorkflowStream] Connection confirmed');
              break;
              
            case 'subtask_update':
              if (onSubtaskUpdate && data.data) {
                onSubtaskUpdate(data.data as SubtaskUpdate);
              }
              break;
              
            case 'execution_log':
              if (onLog && data.data) {
                onLog(data.data);
              }
              break;
              
            case 'workflow_complete':
              console.log('✅ [useWorkflowStream] Workflow completed');
              if (onWorkflowComplete && data.data) {
                onWorkflowComplete(data.data);
              }
              // Auto-disconnect on completion
              if (autoDisconnectTimeoutRef.current) {
                clearTimeout(autoDisconnectTimeoutRef.current);
              }
              autoDisconnectTimeoutRef.current = setTimeout(() => disconnect(), 1000);
              break;
              
            case 'error':
            {
              console.error('❌ [useWorkflowStream] Server error:', data);
              const errorMsg = data.data?.message || 'Server error';
              setError(errorMsg);
              if (onError) {
                onError(errorMsg);
              }
              break;
            }
              
            default:
              console.warn('[useWorkflowStream] Unknown event type:', data.type);
          }
        } catch (err) {
          console.error('[useWorkflowStream] Failed to parse SSE event:', err);
        }
      };

      es.onerror = (err) => {
        console.error('❌ [useWorkflowStream] SSE error:', err);
        setIsConnected(false);
        
        es.close();
        eventSourceRef.current = null;
        
        // Auto-reconnect if enabled
        if (autoReconnect && shouldConnectRef.current) {
          console.log(`🔄 [useWorkflowStream] Reconnecting in ${reconnectDelay}ms...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        } else {
          setError('Connection lost');
        }
      };
      
    } catch (err) {
      console.error('[useWorkflowStream] Failed to create EventSource:', err);
      setError(err instanceof Error ? err.message : 'Connection failed');
    }
  }, [executionId, onSubtaskUpdate, onWorkflowComplete, onLog, onError, autoReconnect, reconnectDelay, disconnect]);

  const reconnect = useCallback(() => {
    disconnect();
    shouldConnectRef.current = true;
    setTimeout(() => connect(), 100);
  }, [connect, disconnect]);

  // Auto-connect when executionId changes
  useEffect(() => {
    if (executionId) {
      shouldConnectRef.current = true;
      connect();
    } else {
      disconnect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [executionId, connect, disconnect]);

  return {
    isConnected,
    error,
    reconnect,
    disconnect,
  };
}

