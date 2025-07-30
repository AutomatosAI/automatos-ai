const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.automatos.app';

export interface SystemHealth {
  status: string;
  timestamp: string;
  version: string;
}

export interface SystemMetrics {
  cpu: {
    usage_percent: number;
    cores: number;
  };
  memory: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  disk: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  network: {
    packets_sent: number;
    packets_recv: number;
    bytes_sent: number;
    bytes_recv: number;
  };
  timestamp: number;
}

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  async getSystemHealth(): Promise<SystemHealth> {
    return this.request<SystemHealth>('/health');
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.request<SystemMetrics>('/api/system/metrics');
  }
}

const apiClient = new ApiClient();
export default apiClient;
