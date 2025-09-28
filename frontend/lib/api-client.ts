/**
 * API Client for Automatos
 */

const logger = require('./logger');

class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = 'http://206.81.0.227:8000';
    logger.info('API Client initialized with baseUrl:', this.baseUrl);
  }

  async getSystemHealth() {
    return { status: 'ok', message: 'System is healthy' };
  }

  async getAgents() {
    return [
      { id: '1', name: 'Agent 1', status: 'active' },
      { id: '2', name: 'Agent 2', status: 'idle' }
    ];
  }

  async getDocuments() {
    return [
      { id: '1', name: 'Document 1' },
      { id: '2', name: 'Document 2' }
    ];
  }

  async getWorkflows() {
    return [
      { id: '1', name: 'Workflow 1' },
      { id: '2', name: 'Workflow 2' }
    ];
  }
}

// Export a singleton instance
const apiClient = new ApiClient();
export { apiClient };
export default apiClient;
