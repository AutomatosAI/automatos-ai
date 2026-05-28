
/**
 * React Hook for Skills API
 * =========================
 * 
 * Custom hook for interacting with PRD-22 Skills API endpoints.
 */

import { useState, useCallback } from 'react';
import { useToast } from '@/hooks/use-toast';
import {
  Skill,
  SkillSource,
  GitImportRequest,
  GitImportResponse,
  SkillContentResponse,
  SkillRecommendation,
  SkillSourceType,
  SkillSourceStatus
} from '@/types/skills';
import { apiClient } from '@/lib/api-client';

function getSkillsApiBase(): string {
  const base = apiClient.getBaseUrl();
  return base ? `${base}/api/v1/skills` : '/api/v1/skills';
}

function getWorkspaceSkillsBase(workspaceId: string): string {
  const base = apiClient.getBaseUrl();
  const root = base ? `${base}/api/workspaces` : '/api/workspaces';
  return `${root}/${workspaceId}/skills`;
}

export interface ScannerFinding {
  type: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | string;
  line: number;
  description: string;
  matched_text: string;
}

export interface WorkspaceSkillContent {
  skill_id: number;
  name: string;
  description: string | null;
  category: string | null;
  tags: string[];
  skill_version: string | null;
  skill_source: string | null;
  content: string;
  origin: 'workspace' | 'marketplace';
  forked_from_skill_id: number | null;
  editable: boolean;
}

export interface CreateSkillPayload {
  name: string;
  content: string;
  description?: string;
  category?: string;
  tags?: string[];
  skill_type?: string;
  acknowledge_warnings?: boolean;
}

export interface UpdateSkillPayload {
  content: string;
  description?: string;
  category?: string;
  tags?: string[];
  acknowledge_warnings?: boolean;
}

export interface SkillSaveResult {
  success: true;
  skill_id: number;
  forked?: boolean;
  forked_from_skill_id?: number;
  agents_migrated?: number;
  warnings: ScannerFinding[];
}

export class SkillScanError extends Error {
  status: 'blocked' | 'warnings';
  findings: ScannerFinding[];
  constructor(status: 'blocked' | 'warnings', message: string, findings: ScannerFinding[]) {
    super(message);
    this.name = 'SkillScanError';
    this.status = status;
    this.findings = findings;
  }
}

async function fetchWithAuth(url: string, init?: RequestInit): Promise<Response> {
  const auth = await apiClient.getAuthHeaders();
  return fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...auth, ...(init?.headers as Record<string, string>) }
  });
}

export function useSkillsApi() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // =========================================================================
  // Skill Source Management
  // =========================================================================

  const importGitRepository = useCallback(async (data: GitImportRequest): Promise<GitImportResponse | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/sources/git`, {
        method: 'POST',
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        let detail = 'Failed to import Git repository';
        const text = await response.text();
        if (text) {
          try {
            const errorData = JSON.parse(text);
            detail = errorData.detail || (typeof errorData.error === 'string' ? errorData.error : detail);
          } catch (_) {
            detail = text.slice(0, 200);
          }
        }
        throw new Error(detail);
      }

      const result = await response.json();
      
      toast({
        title: 'Success',
        description: `Imported ${result.skills_discovered} skills from ${result.source_name}`,
        variant: 'default'
      });

      return result;
    } catch (err: any) {
      const errorMsg = err.message || 'Unknown error';
      setError(errorMsg);
      
      toast({
        title: 'Error',
        description: errorMsg,
        variant: 'destructive'
      });
      
      return null;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const listSkillSources = useCallback(async (
    sourceType?: SkillSourceType,
    status?: SkillSourceStatus
  ): Promise<SkillSource[]> => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      if (sourceType) params.append('source_type', sourceType);
      if (status) params.append('status', status);

      const response = await fetchWithAuth(`${getSkillsApiBase()}/sources?${params}`);

      if (!response.ok) {
        throw new Error('Failed to fetch skill sources');
      }

      const sources = await response.json();
      return sources;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Error',
        description: err.message,
        variant: 'destructive'
      });
      return [];
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const getSkillSource = useCallback(async (sourceId: number): Promise<SkillSource | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/sources/${sourceId}`);

      if (!response.ok) {
        throw new Error('Failed to fetch skill source');
      }

      const source = await response.json();
      return source;
    } catch (err: any) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateGitRepository = useCallback(async (sourceId: number): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/sources/${sourceId}/update`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to update repository');
      }

      const result = await response.json();
      
      toast({
        title: 'Repository Updated',
        description: result.message,
        variant: 'default'
      });

      return true;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Update Failed',
        description: err.message,
        variant: 'destructive'
      });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const rollbackGitRepository = useCallback(async (
    sourceId: number,
    commitSha: string
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(
        `${getSkillsApiBase()}/sources/${sourceId}/rollback?commit_sha=${commitSha}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to rollback repository');
      }

      toast({
        title: 'Repository Rolled Back',
        description: `Rolled back to commit ${commitSha}`,
        variant: 'default'
      });

      return true;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Rollback Failed',
        description: err.message,
        variant: 'destructive'
      });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const deactivateSkillSource = useCallback(async (
    sourceId: number,
    deactivateSkills: boolean = false
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(
        `${getSkillsApiBase()}/sources/${sourceId}?deactivate_skills=${deactivateSkills}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        throw new Error('Failed to deactivate skill source');
      }

      toast({
        title: 'Source Deactivated',
        description: 'Skill source has been deactivated',
        variant: 'default'
      });

      return true;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Error',
        description: err.message,
        variant: 'destructive'
      });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // =========================================================================
  // Skill Management
  // =========================================================================

  const listSkills = useCallback(async (params?: {
    category?: string;
    source?: string;
    search?: string;
    tags?: string;
    limit?: number;
    offset?: number;
  }): Promise<Skill[]> => {
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = new URLSearchParams();
      if (params?.category) queryParams.append('category', params.category);
      if (params?.source) queryParams.append('source', params.source);
      if (params?.search) queryParams.append('search', params.search);
      if (params?.tags) queryParams.append('tags', params.tags);
      if (params?.limit) queryParams.append('limit', params.limit.toString());
      if (params?.offset) queryParams.append('offset', params.offset.toString());

      const response = await fetchWithAuth(`${getSkillsApiBase()}?${queryParams}`);

      if (!response.ok) {
        throw new Error('Failed to fetch skills');
      }

      const skills = await response.json();
      return skills;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Error',
        description: err.message,
        variant: 'destructive'
      });
      return [];
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const getSkillDetails = useCallback(async (skillId: number): Promise<Skill | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/${skillId}`);

      if (!response.ok) {
        throw new Error('Failed to fetch skill details');
      }

      const skill = await response.json();
      return skill;
    } catch (err: any) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const getSkillContent = useCallback(async (
    skillId: number,
    level: number = 2,
    resourcePath?: string
  ): Promise<SkillContentResponse | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({ level: level.toString() });
      if (resourcePath) params.append('resource_path', resourcePath);

      const response = await fetchWithAuth(`${getSkillsApiBase()}/${skillId}/content?${params}`);

      if (!response.ok) {
        throw new Error('Failed to fetch skill content');
      }

      const content = await response.json();
      return content;
    } catch (err: any) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const deactivateSkill = useCallback(async (skillId: number): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/${skillId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to deactivate skill');
      }

      toast({
        title: 'Skill Deactivated',
        description: 'Skill has been deactivated',
        variant: 'default'
      });

      return true;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Error',
        description: err.message,
        variant: 'destructive'
      });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // =========================================================================
  // Agent-Skill Assignment
  // =========================================================================

  const getAgentSkills = useCallback(async (agentId: number): Promise<Skill[]> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/agents/${agentId}/skills`);

      if (!response.ok) {
        throw new Error('Failed to fetch agent skills');
      }

      const skills = await response.json();
      return skills;
    } catch (err: any) {
      setError(err.message);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const assignSkillsToAgent = useCallback(async (
    agentId: number,
    skillIds: number[],
    replace: boolean = false
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(
        `${getSkillsApiBase()}/agents/${agentId}/skills?replace=${replace}`,
        {
          method: 'POST',
          body: JSON.stringify(skillIds)
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to assign skills');
      }

      toast({
        title: 'Skills Assigned',
        description: `Successfully assigned ${skillIds.length} skills to agent`,
        variant: 'default'
      });

      return true;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Assignment Failed',
        description: err.message,
        variant: 'destructive'
      });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const removeSkillsFromAgent = useCallback(async (
    agentId: number,
    skillIds: number[]
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = skillIds.map(id => `skill_ids=${id}`).join('&');
      const response = await fetchWithAuth(
        `${getSkillsApiBase()}/agents/${agentId}/skills?${queryParams}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        throw new Error('Failed to remove skills');
      }

      toast({
        title: 'Skills Removed',
        description: `Removed ${skillIds.length} skills from agent`,
        variant: 'default'
      });

      return true;
    } catch (err: any) {
      setError(err.message);
      toast({
        title: 'Error',
        description: err.message,
        variant: 'destructive'
      });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // =========================================================================
  // Skill Recommendations
  // =========================================================================

  const recommendSkills = useCallback(async (
    taskDescription: string,
    taskType?: string,
    limit: number = 5
  ): Promise<SkillRecommendation[]> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetchWithAuth(`${getSkillsApiBase()}/recommend`, {
        method: 'POST',
        body: JSON.stringify({
          task_description: taskDescription,
          task_type: taskType,
          limit
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get skill recommendations');
      }

      const recommendations = await response.json();
      return recommendations;
    } catch (err: any) {
      setError(err.message);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  // =========================================================================
  // Workspace Skill Editor (PRD: User-editable workspace skills)
  // =========================================================================

  const listWorkspaceSkills = useCallback(async (workspaceId: string) => {
    setError(null);
    const response = await fetchWithAuth(`${getWorkspaceSkillsBase(workspaceId)}`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || 'Failed to load workspace skills');
    }
    const json = await response.json();
    return {
      items: (json.items ?? []) as Array<{
        skill_id: number;
        name: string;
        description: string | null;
        category: string | null;
        skill_version: string | null;
        tags: string[] | null;
        estimated_tokens: number;
        skill_source: string | null;
        enabled_at: string | null;
        origin: 'marketplace' | 'workspace';
        forked_from_skill_id: number | null;
        assigned_agent_count: number;
      }>,
      marketplace_total: (json.marketplace_total ?? 0) as number,
    };
  }, []);

  const getWorkspaceSkillContent = useCallback(async (
    workspaceId: string,
    skillId: number,
  ): Promise<WorkspaceSkillContent> => {
    setError(null);
    const response = await fetchWithAuth(`${getWorkspaceSkillsBase(workspaceId)}/${skillId}/content`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || 'Failed to load skill content');
    }
    return response.json();
  }, []);

  const _parseSaveResponse = async (response: Response): Promise<SkillSaveResult> => {
    if (response.ok) {
      return response.json();
    }
    const text = await response.text();
    let parsed: any = null;
    try { parsed = JSON.parse(text); } catch { /* not JSON */ }
    const detail = parsed?.detail;
    if (response.status === 422 && detail && typeof detail === 'object' && 'status' in detail) {
      throw new SkillScanError(
        detail.status,
        detail.message || 'Skill failed security scan',
        detail.findings || [],
      );
    }
    throw new Error(typeof detail === 'string' ? detail : (parsed?.message || 'Failed to save skill'));
  };

  const createWorkspaceSkill = useCallback(async (
    workspaceId: string,
    payload: CreateSkillPayload,
  ): Promise<SkillSaveResult> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWithAuth(`${getWorkspaceSkillsBase(workspaceId)}/create`, {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      const result = await _parseSaveResponse(response);
      toast({
        title: 'Skill created',
        description: `'${payload.name}' added to your workspace.`,
      });
      return result;
    } catch (err: any) {
      if (!(err instanceof SkillScanError)) {
        setError(err.message);
        toast({ title: 'Save failed', description: err.message, variant: 'destructive' });
      }
      throw err;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const updateWorkspaceSkill = useCallback(async (
    workspaceId: string,
    skillId: number,
    payload: UpdateSkillPayload,
  ): Promise<SkillSaveResult> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWithAuth(`${getWorkspaceSkillsBase(workspaceId)}/${skillId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      });
      const result = await _parseSaveResponse(response);
      toast({
        title: result.forked ? 'Skill forked & saved' : 'Skill saved',
        description: result.forked
          ? `Created your workspace copy. ${result.agents_migrated ?? 0} agent assignment(s) migrated.`
          : 'Your changes are live.',
      });
      return result;
    } catch (err: any) {
      if (!(err instanceof SkillScanError)) {
        setError(err.message);
        toast({ title: 'Save failed', description: err.message, variant: 'destructive' });
      }
      throw err;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const deleteWorkspaceSkill = useCallback(async (
    workspaceId: string,
    skillId: number,
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWithAuth(`${getWorkspaceSkillsBase(workspaceId)}/${skillId}/owned`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to delete skill');
      }
      toast({ title: 'Skill deleted' });
      return true;
    } catch (err: any) {
      setError(err.message);
      toast({ title: 'Delete failed', description: err.message, variant: 'destructive' });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const disableWorkspaceSkill = useCallback(async (
    workspaceId: string,
    skillId: number,
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWithAuth(`${getWorkspaceSkillsBase(workspaceId)}/${skillId}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to remove skill from workspace');
      }
      toast({ title: 'Skill removed from workspace' });
      return true;
    } catch (err: any) {
      setError(err.message);
      toast({ title: 'Remove failed', description: err.message, variant: 'destructive' });
      return false;
    } finally {
      setLoading(false);
    }
  }, [toast]);

  return {
    loading,
    error,
    // Source management
    importGitRepository,
    listSkillSources,
    getSkillSource,
    updateGitRepository,
    rollbackGitRepository,
    deactivateSkillSource,
    // Skill management
    listSkills,
    getSkillDetails,
    getSkillContent,
    deactivateSkill,
    // Agent-skill assignment
    getAgentSkills,
    assignSkillsToAgent,
    removeSkillsFromAgent,
    // Recommendations
    recommendSkills,
    // Workspace skill editor
    listWorkspaceSkills,
    getWorkspaceSkillContent,
    createWorkspaceSkill,
    updateWorkspaceSkill,
    deleteWorkspaceSkill,
    disableWorkspaceSkill,
  };
}
