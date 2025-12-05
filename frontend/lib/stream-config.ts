/**
 * Workflow Streaming Configuration
 * 
 * Centralized configuration for AI SDK streaming endpoints and settings.
 */

export const STREAM_CONFIG = {
    // API Endpoints
    endpoints: {
        workflowStream: '/api/workflows/stream',
        executionStream: (executionId: number) => `/api/workflows/executions/${executionId}/stream`,
        executionStreamAISDK: (executionId: number) => `/api/workflows/executions/${executionId}/stream/aisdk`,
    },

    // Retry Configuration
    retry: {
        maxRetries: 3,
        retryDelay: 1000, // ms
        backoffMultiplier: 2,
    },

    // Timeout Settings
    timeouts: {
        connectionTimeout: 30000, // 30s
        heartbeatInterval: 15000, // 15s
        maxIdleTime: 60000, // 60s
    },

    // Event Types
    eventTypes: {
        // Connection events
        CONNECTED: 'connected',
        HEARTBEAT: 'heartbeat',

        // Stage events
        STAGE_START: 'stage_start',
        STAGE_COMPLETE: 'stage_complete',

        // Execution events
        EXECUTION_LOG: 'execution_log',
        SUBTASK_UPDATE: 'subtask_update',
        WORKFLOW_COMPLETE: 'workflow_complete',

        // Error events
        ERROR: 'error',
    },

    // Stage Names (9-stage workflow)
    stages: {
        1: 'Task Decomposition',
        2: 'Agent Selection',
        3: 'Context Engineering',
        4: 'Agent Execution',
        5: 'Result Aggregation',
        6: 'Learning Update',
        7: 'Quality Assessment',
        8: 'Memory Storage',
        9: 'Response Generation',
    },

    // Log Levels
    logLevels: {
        DEBUG: 'DEBUG',
        INFO: 'INFO',
        WARNING: 'WARNING',
        ERROR: 'ERROR',
    },
} as const;

export type StreamEventType = typeof STREAM_CONFIG.eventTypes[keyof typeof STREAM_CONFIG.eventTypes];
export type LogLevel = typeof STREAM_CONFIG.logLevels[keyof typeof STREAM_CONFIG.logLevels];
export type StageNumber = keyof typeof STREAM_CONFIG.stages;
