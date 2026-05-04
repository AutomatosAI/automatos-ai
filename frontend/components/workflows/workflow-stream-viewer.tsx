'use client';

import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Loader2, CheckCircle, AlertCircle, Clock, Zap, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface WorkflowStreamViewerProps {
    executionId: number;
    className?: string;
    onStageUpdate?: (stage: number, stageName: string) => void;
    onComplete?: () => void;
}

interface StageData {
    stage: number;
    stage_name: string;
    status?: string;
    duration_ms?: number;
    result?: any;
}

interface LogData {
    level: string;
    message: string;
    details?: any;
    timestamp?: string;
}

interface StreamMessage {
    id: string;
    content: string;
    timestamp: Date;
}

interface StreamDataEvent {
    type: string;
    data?: any;
    timestamp?: string;
}

export function WorkflowStreamViewer({
    executionId,
    className,
    onStageUpdate,
    onComplete
}: WorkflowStreamViewerProps) {
    const [currentStage, setCurrentStage] = useState<StageData | null>(null);
    const [isComplete, setIsComplete] = useState(false);
    const [messages, setMessages] = useState<StreamMessage[]>([]);
    const [dataEvents, setDataEvents] = useState<StreamDataEvent[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const eventSourceRef = useRef<EventSource | null>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (autoScroll && scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, dataEvents, autoScroll]);

    // Connect to AI SDK stream
    useEffect(() => {
        if (!executionId) return;

        console.log('🚀 Starting AI SDK stream for execution:', executionId);
        setIsLoading(true);
        setError(null);
        setMessages([]);
        setDataEvents([]);
        setCurrentStage(null);
        setIsComplete(false);

        // Create AbortController for fetch cancellation
        const abortController = new AbortController();
        let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

        // Use fetch with streaming for AI SDK protocol
        const connectStream = async () => {
            try {
                const response = await fetch('/api/workflows/stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ executionId }),
                    signal: abortController.signal, // Pass abort signal to fetch
                });

                if (!response.ok) {
                    throw new Error(`Stream failed: ${response.statusText}`);
                }

                if (!response.body) {
                    throw new Error('No response body');
                }

                reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        console.log('✅ Stream complete');
                        setIsLoading(false);
                        break;
                    }

                    // Decode chunk and add to buffer
                    buffer += decoder.decode(value, { stream: true });

                    // Process complete lines
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || ''; // Keep incomplete line in buffer

                    for (const line of lines) {
                        if (!line.trim()) continue;

                        try {
                            // Parse AI SDK protocol
                            if (line.startsWith('0:')) {
                                // Text delta - add to messages
                                const text = JSON.parse(line.slice(2));
                                setMessages(prev => [...prev, {
                                    id: `msg-${Date.now()}-${Math.random()}`,
                                    content: text,
                                    timestamp: new Date()
                                }]);
                            } else if (line.startsWith('d:')) {
                                // Data payload - add to data events
                                const data = JSON.parse(line.slice(2));
                                setDataEvents(prev => [...prev, data]);

                                // Process events IMMEDIATELY to avoid missing rapid updates
                                if (data?.type === 'stage_start') {
                                    const stageData: StageData = {
                                        stage: data.data?.stage || 0,
                                        stage_name: data.data?.stage_name || 'Unknown',
                                        status: 'running'
                                    };
                                    setCurrentStage(stageData);
                                    onStageUpdate?.(stageData.stage, stageData.stage_name);
                                } else if (data?.type === 'stage_complete') {
                                    const stageData: StageData = {
                                        stage: data.data?.stage || 0,
                                        stage_name: data.data?.stage_name || 'Unknown',
                                        status: 'complete',
                                        duration_ms: data.data?.duration_ms,
                                        result: data.data?.result
                                    };
                                    setCurrentStage(stageData);
                                } else if (data?.type === 'workflow_complete') {
                                    setIsComplete(true);
                                    setIsLoading(false);
                                    onComplete?.();
                                }

                            } else if (line.startsWith('e:')) {
                                // Error event
                                const errorData = JSON.parse(line.slice(2));
                                setError(new Error(errorData.message || 'Stream error'));
                                setIsLoading(false);
                            }
                        } catch (err) {
                            console.error('Error parsing stream line:', err, line);
                        }
                    }
                }
            } catch (err: any) {
                // Ignore AbortError - it's expected when component unmounts
                if (err.name === 'AbortError') {
                    console.log('🛑 Stream aborted (component unmounted or cancelled)');
                    return;
                }

                console.error('❌ Stream error:', err);
                setError(err);
                setIsLoading(false);
            }
        };

        connectStream();

        return () => {
            // Cleanup on unmount or executionId change
            console.log('🧹 Cleaning up stream connection');

            // Abort the fetch request
            abortController.abort();

            // Cancel the reader if it exists
            reader?.cancel().catch(err => {
                // Ignore errors during cleanup
                console.log('Reader cancel error (safe to ignore):', err.message);
            });

            // Reset loading state
            setIsLoading(false);
        };
    }, [executionId]);

    // Get log level styling
    const getLogLevelStyle = (level: string) => {
        switch (level?.toUpperCase()) {
            case 'ERROR':
                return 'text-destructive border-destructive/30 bg-destructive/10';
            case 'WARNING':
                return 'text-warning border-warning/30 bg-warning/10';
            case 'DEBUG':
                return 'text-info border-info/30 bg-info/10';
            default:
                return 'text-success border-success/30 bg-success/10';
        }
    };

    // Parse data events for structured display
    const structuredLogs = dataEvents.filter((d: any) =>
        d?.type === 'execution_log' || d?.type === 'subtask_update'
    );

    return (
        <Card className={cn('w-full h-[600px] flex flex-col', className)}>
            <CardHeader className="flex flex-row items-center justify-between py-3 border-b border-border/30">
                <div className="flex items-center gap-3">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Zap className="h-4 w-4 text-primary" />
                        Live Execution Stream
                        {isLoading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
                    </CardTitle>

                    {currentStage && (
                        <Badge
                            variant={currentStage.status === 'complete' ? 'default' : 'secondary'}
                            className={cn(
                                'text-xs',
                                currentStage.status === 'running' && 'bg-primary/20 text-primary animate-pulse'
                            )}
                        >
                            Stage {currentStage.stage}: {currentStage.stage_name}
                        </Badge>
                    )}

                    {isComplete && (
                        <Badge variant="default" className="bg-success/20 text-success border-success/30">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Complete
                        </Badge>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setAutoScroll(!autoScroll)}
                        className={cn(
                            'text-xs',
                            autoScroll ? 'text-success' : 'text-muted-foreground'
                        )}
                    >
                        {autoScroll ? '⬇ Auto-scroll ON' : '⬇ Auto-scroll OFF'}
                    </Button>
                </div>
            </CardHeader>

            <CardContent className="flex-1 p-0 overflow-hidden">
                <ScrollArea className="h-full p-4" ref={scrollRef}>
                    {error && (
                        <div className="text-destructive text-sm mb-4 p-3 bg-destructive/10 border border-destructive/30 rounded-lg flex items-start gap-2">
                            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                            <div>
                                <div className="font-semibold">Stream Error</div>
                                <div className="text-xs mt-1">{error.message}</div>
                            </div>
                        </div>
                    )}

                    <div className="space-y-2">
                        {/* Text messages (stage updates, progress) */}
                        {messages.map((m) => (
                            <div
                                key={m.id}
                                className="text-sm font-mono whitespace-pre-wrap p-2 rounded-lg bg-white/5 border border-border/30"
                            >
                                {m.content}
                            </div>
                        ))}

                        {/* Structured data events (logs, metrics) */}
                        {structuredLogs.map((d: any, i: number) => {
                            const isLog = d?.type === 'execution_log';
                            const isSubtask = d?.type === 'subtask_update';

                            if (isLog) {
                                const logData: LogData = d.data || {};
                                return (
                                    <div
                                        key={i}
                                        className={cn(
                                            'text-xs border-l-2 pl-3 py-2 rounded-r-lg',
                                            getLogLevelStyle(logData.level)
                                        )}
                                    >
                                        <div className="flex items-center gap-2 mb-1">
                                            <Badge variant="outline" className="text-[10px] h-4">
                                                {logData.level || 'INFO'}
                                            </Badge>
                                            {logData.timestamp && (
                                                <span className="text-[10px] text-muted-foreground font-mono">
                                                    {new Date(logData.timestamp).toLocaleTimeString()}
                                                </span>
                                            )}
                                        </div>
                                        <div className="font-medium">{logData.message}</div>
                                        {logData.details && (
                                            <details className="mt-2">
                                                <summary className="cursor-pointer text-[10px] text-muted-foreground hover:text-foreground">
                                                    View details
                                                </summary>
                                                <pre className="mt-2 text-[10px] bg-black/30 p-2 rounded overflow-x-auto">
                                                    {JSON.stringify(logData.details, null, 2)}
                                                </pre>
                                            </details>
                                        )}
                                    </div>
                                );
                            }

                            if (isSubtask) {
                                const subtaskData = d.data || {};
                                const status = subtaskData.status || 'running';
                                const statusIcon = status === 'completed' ? '✓' : status === 'failed' ? '✗' : '⋯';
                                const statusColor = status === 'completed' ? 'text-success' : status === 'failed' ? 'text-destructive' : 'text-info';

                                return (
                                    <div
                                        key={i}
                                        className="text-xs p-2 rounded-lg bg-white/5 border border-border/30"
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className={cn('font-bold', statusColor)}>{statusIcon}</span>
                                            <span className="font-medium">{subtaskData.description || 'Task'}</span>
                                            <Badge variant="outline" className="text-[10px] h-4 ml-auto">
                                                {status}
                                            </Badge>
                                        </div>
                                    </div>
                                );
                            }

                            return null;
                        })}

                        {/* Loading indicator */}
                        {isLoading && !isComplete && (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground p-2">
                                <Loader2 className="h-3 w-3 animate-spin" />
                                <span>Streaming updates...</span>
                            </div>
                        )}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
