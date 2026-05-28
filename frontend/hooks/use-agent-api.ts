
'use client'

/**
 * Enhanced Agent API hooks for real-time data fetching and mutations
 * Provides React Query integration with automatic caching, retries, and real-time updates
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'

// API client - you'll need to adjust the import path based on your project structure
const API_BASE = process.env.NEXT_PUBLIC_API_URL || ''
const API_PREFIX = "/api"

// Simple API client for demonstration - replace with your actual API client
// Import real API client
import { apiClient } from "@/lib/api-client"

// Agent API client using main API client
const agentApiClient = {
  // Agent endpoints
  getAgents: () => apiClient.getAgents(),
  getAgent: (id: string) => apiClient.getAgent(id),
  getAgentStats: () => apiClient.getSystemAgentStatistics(),
  getAgentTypes: () => apiClient.getSystemAgentTypes(),
  createAgent: (data: any) => apiClient.createAgent(data),
  updateAgent: (id: string, data: any) => apiClient.updateAgent(id, data),
  deleteAgent: (id: string) => apiClient.deleteAgent(id),

  // Skills endpoints
  getSkills: () => apiClient.getSkills(),
  getAgentSkills: (agentId: string) => apiClient.getAgentSkills(agentId),
  addSkillToAgent: (agentId: string, skillId: string) => apiClient.addSkillToAgent(agentId, skillId),
  removeSkillFromAgent: (agentId: string, skillId: string) => apiClient.removeSkillFromAgent(agentId, skillId),
  createSkill: (data: any) => apiClient.createSkill(data),
  updateSkill: (id: string, data: any) => apiClient.updateSkill(id, data),
  deleteSkill: (id: string) => apiClient.deleteSkill(id),
  createSkillsBulk: (data: any[]) => apiClient.createSkillsBulk(data),

  // Configuration endpoints
  getAgentConfig: (agentId: string) => apiClient.getAgent(agentId),
  updateAgentConfig: (agentId: string, config: any) => apiClient.updateAgent(agentId, config),

  // Coordination endpoints
  getAgentCoordination: () => apiClient.coordinateAgents({ agents: [], task: {} }),
  coordinateAgents: (data: any) => apiClient.coordinateAgents(data),
  collaborativeReasoning: (data: any) => apiClient.collaborativeReasoning(data),

  // Performance endpoints
  getAgentPerformance: (agentId: string) => apiClient.getAgentPerformance(agentId),
  getAgentLogs: (agentId: string) => apiClient.getAgentLogs(agentId),
  getAgentMetrics: (agentId: string) => apiClient.getAgentStats(agentId),

  // Execution endpoints
  startAgent: (agentId: string) => apiClient.startAgent(agentId),
  stopAgent: (agentId: string) => apiClient.stopAgent(agentId),
  pauseAgent: (agentId: string) => apiClient.stopAgent(agentId), // No pause endpoint, use stop
}

// Query keys for React Query
export const agentQueryKeys = {
  // Agents
  agents: ['agents'] as const,
  agent: (id: string) => ['agents', id] as const,
  agentStats: ['agents', 'stats'] as const,
  agentTypes: ['agents', 'types'] as const,
  agentLogs: (id: string) => ['agents', id, 'logs'] as const,
  agentMetrics: (id: string) => ['agents', id, 'metrics'] as const,
  agentPerformance: (id: string) => ['agents', id, 'performance'] as const,

  // Skills
  skills: ['skills'] as const,
  agentSkills: (agentId: string) => ['agents', agentId, 'skills'] as const,

  // Configuration
  agentConfig: (agentId: string) => ['agents', agentId, 'configuration'] as const,

  // Coordination
  coordination: ['coordination'] as const,
  coordinationStatus: ['coordination', 'status'] as const,
}

// ============= QUERY HOOKS =============

import { useSystemIcons } from './use-system-config-api'
import { LEGACY_CATEGORY_MAP } from '@/lib/agent-constants'

interface UseAgentsOptions {
  /** Include the workspace's system agents (e.g. Auto). Used by assignee
   *  pickers — the Roster never sets this. */
  includeWorkspaceSystem?: boolean
}

// Get all agents
export function useAgents(options: UseAgentsOptions = {}) {
  const { data: iconMappings = {} } = useSystemIcons();
  const iconKeys = Object.keys(iconMappings).sort().join(',')
  const includeSystem = options.includeWorkspaceSystem === true

  return useQuery({
    queryKey: [...agentQueryKeys.agents, iconKeys, includeSystem ? 'sys' : 'noSys'],
    queryFn: async () => {
      const agents = (await apiClient.getAgents(0, 100, {
        includeWorkspaceSystem: includeSystem,
      })) as any[];
      // Inject icon mapping based on category (normalize legacy Title Case categories)
      return agents.map((agent: any) => {
        const cat = agent.marketplace_category || agent.configuration?.category || agent.agent_type
        const normalized = LEGACY_CATEGORY_MAP[cat] || cat
        return {
          ...agent,
          premium_icon: iconMappings[normalized] || iconMappings[cat] || null
        }
      });
    },
    refetchInterval: false, // Disable automatic refetching
    staleTime: 30000, // Cache for 30 seconds, allows manual refresh
    refetchOnWindowFocus: false, // Don't refetch on window focus
  })
}

/**
 * Variant of useAgents() that includes the workspace's hidden Auto so it
 * can be selected as a task assignee. Use this in board pickers, task
 * dialogs, and agent dropdowns where Auto should appear; never in the
 * Roster (which deliberately hides it).
 */
export function useAssignableAgents() {
  return useAgents({ includeWorkspaceSystem: true })
}

// Get single agent
export function useAgent(agentId: string | null) {
  const { data: iconMappings = {} } = useSystemIcons();
  const iconKeys = Object.keys(iconMappings).sort().join(',')

  return useQuery({
    queryKey: [...agentQueryKeys.agent(agentId!), iconKeys],
    queryFn: async () => {
      const agent = await agentApiClient.getAgent(agentId!);
      const cat = agent.marketplace_category || agent.configuration?.category || agent.agent_type
      const normalized = LEGACY_CATEGORY_MAP[cat] || cat
      return {
        ...agent,
        premium_icon: iconMappings[normalized] || iconMappings[cat] || null
      };
    },
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get agent statistics
export function useAgentStats() {
  return useQuery({
    queryKey: agentQueryKeys.agentStats,
    queryFn: () => agentApiClient.getAgentStats(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get agent types
export function useAgentTypes() {
  return useQuery({
    queryKey: agentQueryKeys.agentTypes,
    queryFn: () => agentApiClient.getAgentTypes(),
    staleTime: 5 * 60 * 1000, // Agent types don't change often
  })
}

// Get all skills
export function useSkills() {
  const { data: iconMappings = {} } = useSystemIcons();

  return useQuery({
    queryKey: agentQueryKeys.skills,
    queryFn: async () => {
      const skills = await agentApiClient.getSkills();
      return skills.map((skill: any) => ({
        ...skill,
        // Assuming skills have a category or type field we can map
        premium_icon: iconMappings[skill.category] || iconMappings[skill.type] || null
      }));
    },
    staleTime: 2 * 60 * 1000, // Skills don't change very often
  })
}

// Get agent skills
export function useAgentSkills(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentSkills(agentId!),
    queryFn: () => agentApiClient.getAgentSkills(agentId!),
    enabled: !!agentId,
    refetchInterval: 30000,
  })
}

// Get agent configuration
export function useAgentConfig(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentConfig(agentId!),
    queryFn: () => agentApiClient.getAgentConfig(agentId!),
    enabled: !!agentId,
  })
}

// Get agent performance
export function useAgentPerformance(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentPerformance(agentId!),
    queryFn: () => agentApiClient.getAgentPerformance(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get agent logs
export function useAgentLogs(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentLogs(agentId!),
    queryFn: () => agentApiClient.getAgentLogs(agentId!),
    enabled: !!agentId,
    refetchInterval: 5000, // More frequent for logs
  })
}

// Get coordination status
export function useCoordinationStatus() {
  return useQuery({
    queryKey: agentQueryKeys.coordinationStatus,
    queryFn: () => agentApiClient.getAgentCoordination(),
    refetchInterval: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Create agent
export function useCreateAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data) => agentApiClient.createAgent(data),
    onSuccess: (data) => {
      // Invalidate and refetch agent queries
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentStats })
      toast.success('Agent created successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to create agent: ${error.message}`)
      throw error
    },
  })
}

// Update agent
export function useUpdateAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) =>
      agentApiClient.updateAgent(id, data),
    onSuccess: (data, variables) => {
      // Invalidate specific agent and list
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(variables.id) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentStats })
      toast.success('Agent updated successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update agent: ${error.message}`)
      throw error
    },
  })
}

// Delete agent
export function useDeleteAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id) => agentApiClient.deleteAgent(id),
    onSuccess: (data, agentId) => {
      // Remove from cache and invalidate lists
      queryClient.removeQueries({ queryKey: agentQueryKeys.agent(agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentStats })
      toast.success('Agent deleted successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to delete agent: ${error.message}`)
      throw error
    },
  })
}

// Add skill to agent
export function useAddSkillToAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ agentId, skillId }: { agentId: string; skillId: string }) =>
      agentApiClient.addSkillToAgent(agentId, skillId),
    onSuccess: (data, variables) => {
      // Invalidate agent skills and agent data
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentSkills(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      return data
    },
    onError: (error) => {
      throw error
    },
  })
}

// Remove skill from agent
export function useRemoveSkillFromAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ agentId, skillId }: { agentId: string; skillId: string }) =>
      agentApiClient.removeSkillFromAgent(agentId, skillId),
    onSuccess: (data, variables) => {
      // Invalidate agent skills and agent data
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentSkills(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      return data
    },
    onError: (error) => {
      throw error
    },
  })
}

// Update agent configuration
export function useUpdateAgentConfig() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ agentId, config }: { agentId: string; config: any }) =>
      agentApiClient.updateAgentConfig(agentId, config),
    onSuccess: (data, variables) => {
      // Invalidate agent config and agent data
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentConfig(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(variables.agentId) })
      // Toast handled by the caller (agent-configuration-modal) to avoid duplicates
      return data
    },
    onError: (error) => {
      throw error
    },
  })
}

// Agent coordination
export function useCoordinateAgents() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data) => agentApiClient.coordinateAgents(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.coordinationStatus })
      toast.success('Agent coordination initiated!')
      return data
    },
    onError: (error) => {
      toast.error(`Coordination failed: ${error.message}`)
      throw error
    },
  })
}

// Collaborative reasoning
export function useCollaborativeReasoning() {
  return useMutation({
    mutationFn: (data) => agentApiClient.collaborativeReasoning(data),
    onSuccess: (data) => {
      toast.success('Collaborative reasoning completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Collaborative reasoning failed: ${error.message}`)
      throw error
    },
  })
}

// Agent execution control
export function useStartAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id) => agentApiClient.startAgent(id),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      toast.success('Agent started successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to start agent: ${error.message}`)
      throw error
    },
  })
}

export function useStopAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id) => agentApiClient.stopAgent(id),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      toast.success('Agent stopped successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to stop agent: ${error.message}`)
      throw error
    },
  })
}

export function usePauseAgent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id) => agentApiClient.pauseAgent(id),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agents })
      toast.success('Agent paused successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to pause agent: ${error.message}`)
      throw error
    },
  })
}


// Create skill
export function useCreateSkill() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (skillData) => agentApiClient.createSkill(skillData),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['skills'] })
      toast.success('Skill created successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to create skill: ${error.message}`)
      throw error
    },
  })
}

// Create skills in bulk
export function useCreateSkillsBulk() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (skillsData) => agentApiClient.createSkillsBulk(skillsData),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['skills'] })
      toast.success(`Created ${data.length} skills successfully!`)
      return data
    },
    onError: (error) => {
      toast.error(`Failed to create skills: ${error.message}`)
      throw error
    },
  })
}

// Update a skill
export function useUpdateSkill() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) => agentApiClient.updateSkill(id, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['skills'] })
      queryClient.invalidateQueries({ queryKey: ['skills', variables.id] })
      toast.success('Skill updated successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update skill: ${error.message}`)
      throw error
    },
  })
}

// Delete a skill
export function useDeleteSkill() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (skillId: string) => agentApiClient.deleteSkill(skillId),
    onSuccess: (data, skillId) => {
      queryClient.invalidateQueries({ queryKey: ['skills'] })
      toast.success('Skill deleted successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to delete skill: ${error.message}`)
      throw error
    },
  })
}
