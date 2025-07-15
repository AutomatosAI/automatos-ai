
'use client';

import { useState, useEffect, useRef } from 'react';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Play,
  AlertCircle,
  Loader2 
} from 'lucide-react';
import { Workflow, ProgressUpdate } from '@/lib/types';
import { apiClient } from '@/lib/api';

interface ProgressPanelProps {
  activeWorkflow: Workflow | null;
}

export function ProgressPanel({ activeWorkflow }: ProgressPanelProps) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!activeWorkflow) {
      setProgress(0);
      setStatus('idle');
      setLogs([]);
      setError(null);
      return;
    }

    // Initialize progress tracking
    setProgress(activeWorkflow.progress);
    setStatus(activeWorkflow.status);
    setError(activeWorkflow.error || null);
    
    if (activeWorkflow.logs) {
      setLogs(activeWorkflow.logs.split('\n').filter(log => log.trim()));
    }

    // Set up WebSocket connection for real-time updates
    const ws = apiClient.createWebSocketConnection(activeWorkflow.id);
    
    if (ws) {
      wsRef.current = ws;

      ws.onopen = () => {
        addLog('Connected to workflow progress stream');
      };

      ws.onmessage = (event) => {
        try {
          const update: ProgressUpdate = JSON.parse(event.data);
          
          if (update.workflowId === activeWorkflow.id) {
            setProgress(update.progress);
            setStatus(update.status);
            
            if (update.message) {
              addLog(update.message);
            }
            
            if (update.error) {
              setError(update.error);
              addLog(`Error: ${update.error}`);
            }
            
            if (update.logs) {
              const newLogs = update.logs.split('\n').filter(log => log.trim());
              newLogs.forEach(log => addLog(log));
            }
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addLog('Connection error - attempting to reconnect...');
      };

      ws.onclose = () => {
        addLog('Connection closed');
        // Attempt to reconnect after a delay
        setTimeout(() => {
          if (activeWorkflow && activeWorkflow.status === 'running') {
            addLog('Reconnecting...');
            // Could implement reconnection logic here
          }
        }, 5000);
      };
    } else {
      // Fallback: poll for updates if WebSocket fails
      addLog('WebSocket unavailable - using polling for updates');
      const pollInterval = setInterval(async () => {
        try {
          const result = await apiClient.getWorkflowStatus(activeWorkflow.id);
          if (result.success && result.data) {
            setProgress(result.data.progress);
            setStatus(result.data.status);
            if (result.data.error) {
              setError(result.data.error);
            }
            if (result.data.status === 'completed' || result.data.status === 'failed') {
              clearInterval(pollInterval);
            }
          }
        } catch (error) {
          console.error('Error polling workflow status:', error);
        }
      }, 2000);

      return () => clearInterval(pollInterval);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [activeWorkflow]);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    
    // Auto-scroll to bottom
    setTimeout(() => {
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }
      }
    }, 100);
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-400" />;
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-400" />;
      default:
        return <Activity className="h-4 w-4 text-slate-400" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'running':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'completed':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'failed':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
    }
  };

  if (!activeWorkflow) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Play className="h-12 w-12 text-slate-600 mb-4" />
        <h3 className="text-lg font-medium text-slate-400 mb-2">No Active Workflow</h3>
        <p className="text-sm text-slate-500">
          Start a new workflow to see real-time progress updates
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status and Progress */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <Badge className={getStatusColor()}>
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </Badge>
          </div>
          <span className="text-sm text-slate-400">{progress}%</span>
        </div>
        
        <Progress 
          value={progress} 
          className="h-2 bg-slate-700"
        />
        
        <div className="text-sm text-slate-400">
          <p className="truncate">
            <span className="font-medium">Repository:</span> {activeWorkflow.githubUrl}
          </p>
          {activeWorkflow.mode === 'prompt-driven' && activeWorkflow.prompt && (
            <p className="mt-1 truncate">
              <span className="font-medium">Task:</span> {activeWorkflow.prompt}
            </p>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert className="border-red-500 bg-red-500/10">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertDescription className="text-red-400">
            {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Activity Logs */}
      <Card className="bg-slate-700/50 border-slate-600">
        <CardContent className="p-4">
          <h4 className="text-sm font-medium text-slate-300 mb-3">Activity Log</h4>
          <ScrollArea className="h-64 w-full" ref={scrollAreaRef}>
            <div className="space-y-1">
              {logs.length > 0 ? (
                logs.map((log, index) => (
                  <div
                    key={index}
                    className="text-xs font-mono text-slate-400 p-2 bg-slate-800/50 rounded border border-slate-600"
                  >
                    {log}
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 italic p-2">
                  No activity logs yet...
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
