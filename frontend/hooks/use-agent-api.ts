
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

// OLD MOCK CLIENT - REPLACED
// /* const apiClient = {
//   async request(endpoint: string, options: RequestInit = {}) {
//     const url = `${API_BASE}${API_PREFIX}${endpoint}`
//     const response = await fetch(url, {
//       headers: {
//         'Content-Type': 'application/json',
//         ...options.headers,
//       },
//       ...options,
//     })
// 
//     if (!response.ok) {
//       throw new Error(`API Error: ${response.status} ${response.statusText}`)
//     }
// 
//     return response.json()
//   },
// 
//   // Agent endpoints
//   getAgents: () => apiClient.request('/agents'),
//   getAgent: (id: string) => apiClient.request(`/agents/${id}`),
//   getAgentStats: () => apiClient.request('/agents/stats'),
//   getAgentTypes: () => apiClient.request('/agents/types'),
//   createAgent: (data: any) => apiClient.request('/agents', {
//     method: 'POST',
//     body: JSON.stringify(data),
//   }),
//   updateAgent: (id: string, data: any) => apiClient.request(`/agents/${id}`, {
//     method: 'PUT',
//     body: JSON.stringify(data),
//   }),
//   deleteAgent: (id: string) => apiClient.request(`/agents/${id}`, {
//     method: 'DELETE',
//   }),
// 
//   // Skills endpoints
//   getSkills: () => apiClient.request('/skills'),
//   getAgentSkills: (agentId: string) => apiClient.request(`/agents/${agentId}/skills`),
//   addSkillToAgent: (agentId: string, skillId: string) => apiClient.request(`/agents/${agentId}/skills`, {
//     method: 'POST',
//     body: JSON.stringify({ skill_id: skillId }),
//   }),
//   removeSkillFromAgent: (agentId: string, skillId: string) => apiClient.request(`/agents/${agentId}/skills/${skillId}`, {
//     method: 'DELETE',
//   }),
// 
//   // Configuration endpoints
//   getAgentConfig: (agentId: string) => apiClient.request(`/agents/${agentId}/configuration`),
//   updateAgentConfig: (agentId: string, config: any) => apiClient.request(`/agents/${agentId}/configuration`, {
//     method: 'PUT',
//     body: JSON.stringify(config),
//   }),
// 
//   // Coordination endpoints
//   getAgentCoordination: () => apiClient.request('/multi-agent/coordination/status'),
//   coordinateAgents: (data: any) => apiClient.request('/multi-agent/coordination/coordinate', {
//     method: 'POST',
//     body: JSON.stringify(data),
//   }),
//   collaborativeReasoning: (data: any) => apiClient.request('/multi-agent/reasoning/collaborative', {
//     method: 'POST',
//     body: JSON.stringify(data),
//   }),
// 
//   // Performance endpoints
//   getAgentPerformance: (agentId: string) => apiClient.request(`/agents/${agentId}/performance`),
//   getAgentLogs: (agentId: string) => apiClient.request(`/agents/${agentId}/logs`),
//   getAgentMetrics: (agentId: string) => apiClient.request(`/agents/${agentId}/metrics`),
// 
//   // Execution endpoints
//   startAgent: (agentId: string) => apiClient.request(`/agents/${agentId}/start`, {
//     method: 'POST',
//   }),
//   stopAgent: (agentId: string) => apiClient.request(`/agents/${agentId}/stop`, {
//     method: 'POST',
//   }),
//   pauseAgent: (agentId: string) => apiClient.request(`/agents/${agentId}/pause`, {
//     method: 'POST',
//   }),
// }
// END MOCK CLIENT - USING REAL API CLIENT NOW */

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

// Get all agents
export function useAgents() {
  return useQuery({
    queryKey: agentQueryKeys.agents,
    queryFn: () => apiClient.getAgents(),
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
  })
}

// Get single agent
export function useAgent(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agent(agentId!),
    queryFn: () => apiClient.getAgent(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get agent statistics
export function useAgentStats() {
  return useQuery({
    queryKey: agentQueryKeys.agentStats,
    queryFn: () => apiClient.getAgentStats(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get agent types
export function useAgentTypes() {
  return useQuery({
    queryKey: agentQueryKeys.agentTypes,
    queryFn: () => apiClient.getAgentTypes(),
    staleTime: 5 * 60 * 1000, // Agent types don't change often
  })
}

// Get all skills
export function useSkills() {
  return useQuery({
    queryKey: agentQueryKeys.skills,
    queryFn: () => apiClient.getSkills(),
    staleTime: 2 * 60 * 1000, // Skills don't change very often
  })
}

// Get agent skills
export function useAgentSkills(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentSkills(agentId!),
    queryFn: () => apiClient.getAgentSkills(agentId!),
    enabled: !!agentId,
    refetchInterval: 30000,
  })
}

// Get agent configuration
export function useAgentConfig(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentConfig(agentId!),
    queryFn: () => apiClient.getAgentConfig(agentId!),
    enabled: !!agentId,
  })
}

// Get agent performance
export function useAgentPerformance(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentPerformance(agentId!),
    queryFn: () => apiClient.getAgentPerformance(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get agent logs
export function useAgentLogs(agentId: string | null) {
  return useQuery({
    queryKey: agentQueryKeys.agentLogs(agentId!),
    queryFn: () => apiClient.getAgentLogs(agentId!),
    enabled: !!agentId,
    refetchInterval: 5000, // More frequent for logs
  })
}

// Get coordination status
export function useCoordinationStatus() {
  return useQuery({
    queryKey: agentQueryKeys.coordinationStatus,
    queryFn: () => apiClient.getAgentCoordination(),
    refetchInterval: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Create agent
export function useCreateAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data) => apiClient.createAgent(data),
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
      apiClient.updateAgent(id, data),
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
    mutationFn: (id) => apiClient.deleteAgent(id),
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
      apiClient.addSkillToAgent(agentId, skillId),
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
      apiClient.removeSkillFromAgent(agentId, skillId),
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
      apiClient.updateAgentConfig(agentId, config),
    onSuccess: (data, variables) => {
      // Invalidate agent config and agent data
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agentConfig(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.agent(variables.agentId) })
      toast.success('Configuration updated successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update configuration: ${error.message}`)
      throw error
    },
  })
}

// Agent coordination
export function useCoordinateAgents() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data) => apiClient.coordinateAgents(data),
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
    mutationFn: (data) => apiClient.collaborativeReasoning(data),
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
    mutationFn: (id) => apiClient.startAgent(id),
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
    mutationFn: (id) => apiClient.stopAgent(id),
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
    mutationFn: (id) => apiClient.pauseAgent(id),
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
    mutationFn: (skillData) => apiClient.createSkill(skillData),
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
    mutationFn: (skillsData) => apiClient.createSkillsBulk(skillsData),
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
    mutationFn: ({ id, data }: { id: string; data: any }) => apiClient.updateSkill(id, data),
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
    mutationFn: (skillId: string) => apiClient.deleteSkill(skillId),
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
