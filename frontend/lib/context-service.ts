/**
 * Context Engineering Service
 * ===========================
 * 
 * Service layer for RAG/context management with real backend integration
 */

import { apiClient, type RAGConfig, type Document } from './api';

export interface ContextStats {
  contextQueries: number;
  retrievalSuccess: number;
  avgResponseTime: string;
  vectorEmbeddings: number;
}

export interface ContextQuery {
  id: string;
  query: string;
  agent: string;
  confidence: number;
  sources: number;
  responseTime: string;
  timestamp: string;
  category: string;
}

export interface ContextPattern {
  id: string;
  name: string;
  description: string;
  usage: number;
  accuracy: number;
  avgSources: number;
  category: string;
  status: 'active' | 'experimental' | 'inactive';
}

export interface ContextSource {
  name: string;
  value: number;
  color: string;
}

export interface RAGPerformanceData {
  time: string;
  queries: number;
  success_rate: number;
  avg_latency: number;
}

class ContextService {
  /**
   * Get context/RAG statistics from real backend data
   */
  async getContextStats(): Promise<ContextStats> {
    try {
      // Use new real context engineering endpoint
      const response = await apiClient.request('/api/context/stats');
      
      return {
        contextQueries: response.contextQueries || 0,
        retrievalSuccess: response.retrievalSuccess || 0,
        avgResponseTime: response.avgResponseTime || '0s',
        vectorEmbeddings: response.vectorEmbeddings || 0
      };
    } catch (error) {
      console.error('Error fetching context stats:', error);
      // Return zero stats if backend is unavailable
      return {
        contextQueries: 0,
        retrievalSuccess: 0,
        avgResponseTime: '0s',
        vectorEmbeddings: 0
      };
    }
  }

  /**
   * Get RAG configurations
   */
  async getRAGConfigs(): Promise<RAGConfig[]> {
    try {
      return await apiClient.getRAGConfigs();
    } catch (error) {
      console.error('Error fetching RAG configs:', error);
      return [];
    }
  }

  /**
   * Get context sources distribution based on documents
   */
  async getContextSources(): Promise<ContextSource[]> {
    try {
      // Use new real context engineering endpoint
      return await apiClient.request('/api/context/sources');
    } catch (error) {
      console.error('Error fetching context sources:', error);
      return [
        { name: 'Loading Error', value: 100, color: '#EF4444' }
      ];
    }
  }

  /**
   * Categorize file types into meaningful context sources
   */
  private categorizeFileType(fileType: string): string {
    const type = fileType.toLowerCase();
    
    if (type.includes('pdf') || type.includes('doc')) {
      return 'Technical Docs';
    } else if (type.includes('md') || type.includes('txt')) {
      return 'Documentation';
    } else if (type.includes('json') || type.includes('yaml')) {
      return 'Configuration';
    } else if (type.includes('py') || type.includes('js') || type.includes('ts')) {
      return 'Code Files';
    } else {
      return 'Other';
    }
  }

  /**
   * Generate recent queries based on real system activity
   */
  async getRecentQueries(): Promise<ContextQuery[]> {
    try {
      // Use new real context engineering endpoint
      return await apiClient.request('/api/context/queries/recent');
    } catch (error) {
      console.error('Error fetching recent queries:', error);
      return [];
    }
  }

  /**
   * Get context patterns based on RAG configurations
   */
  async getContextPatterns(): Promise<ContextPattern[]> {
    try {
      // Use new real context engineering endpoint
      return await apiClient.request('/api/context/patterns');
    } catch (error) {
      console.error('Error fetching context patterns:', error);
      return [];
    }
  }

  /**
   * Generate RAG performance data based on system metrics
   */
  async getRAGPerformanceData(): Promise<RAGPerformanceData[]> {
    try {
      // Use new real context engineering endpoint
      return await apiClient.request('/api/context/performance');
    } catch (error) {
      console.error('Error fetching RAG performance data:', error);
      return [];
    }
  }

  /**
   * Test RAG configuration
   */
  async testRAGConfig(configId: number, query: string): Promise<any> {
    try {
      return await apiClient.testRAGConfig(configId, query);
    } catch (error) {
      console.error('Error testing RAG config:', error);
      throw error;
    }
  }

  /**
   * Create new RAG configuration
   */
  async createRAGConfig(config: {
    name: string;
    embedding_model?: string;
    chunk_size?: number;
    chunk_overlap?: number;
    retrieval_strategy?: string;
    top_k?: number;
    similarity_threshold?: number;
    configuration?: Record<string, any>;
  }): Promise<RAGConfig> {
    return apiClient.createRAGConfig(config);
  }
}

export const contextService = new ContextService();
export default contextService;
