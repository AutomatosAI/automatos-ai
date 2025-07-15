
export interface User {
  id: string;
  email: string;
  name?: string;
  role: string;
}

export interface Workflow {
  id: string;
  githubUrl: string;
  prompt?: string;
  mode: 'self-contained' | 'prompt-driven';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  logs?: string;
  result?: string;
  error?: string;
  userId: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface WorkflowFormData {
  githubUrl: string;
  prompt?: string;
  mode: 'self-contained' | 'prompt-driven';
}

export interface ProgressUpdate {
  workflowId: string;
  status: string;
  progress: number;
  message?: string;
  logs?: string;
  error?: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}
