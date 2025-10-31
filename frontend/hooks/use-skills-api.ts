
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

const API_BASE = '/api/v1/skills';

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
      const response = await fetch(`${API_BASE}/sources/git`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to import Git repository');
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

      const response = await fetch(`${API_BASE}/sources?${params}`);

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
      const response = await fetch(`${API_BASE}/sources/${sourceId}`);

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
      const response = await fetch(`${API_BASE}/sources/${sourceId}/update`, {
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
      const response = await fetch(
        `${API_BASE}/sources/${sourceId}/rollback?commit_sha=${commitSha}`,
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
      const response = await fetch(
        `${API_BASE}/sources/${sourceId}?deactivate_skills=${deactivateSkills}`,
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

      const response = await fetch(`${API_BASE}?${queryParams}`);

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
      const response = await fetch(`${API_BASE}/${skillId}`);

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

      const response = await fetch(`${API_BASE}/${skillId}/content?${params}`);

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
      const response = await fetch(`${API_BASE}/${skillId}`, {
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
      const response = await fetch(`${API_BASE}/agents/${agentId}/skills`);

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
      const response = await fetch(
        `${API_BASE}/agents/${agentId}/skills?replace=${replace}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
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
      const response = await fetch(
        `${API_BASE}/agents/${agentId}/skills?${queryParams}`,
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
      const response = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
    recommendSkills
  };
}
