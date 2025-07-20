
import { WorkflowFormData, ApiResponse, Workflow } from './types';

const API_BASE_URL = 'https://mcp.xplaincrypto.ai';

export class ApiClient {
  private static instance: ApiClient;

  static getInstance(): ApiClient {
    if (!ApiClient.instance) {
      ApiClient.instance = new ApiClient();
    }
    return ApiClient.instance;
  }

  async startWorkflow(data: WorkflowFormData): Promise<ApiResponse<{ workflowId: string }>> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/workflow/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Failed to start workflow');
      }

      return {
        success: true,
        data: result,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  async getWorkflowStatus(workflowId: string): Promise<ApiResponse<Workflow>> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/workflow/${workflowId}`);
      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Failed to get workflow status');
      }

      return {
        success: true,
        data: result,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  createWebSocketConnection(workflowId: string): WebSocket | null {
    try {
      const wsUrl = `wss://mcp.xplaincrypto.ai/ws/${workflowId}`;
      return new WebSocket(wsUrl);
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      return null;
    }
  }
}

export const apiClient = ApiClient.getInstance();
