
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { getDefaultModelConfig, LLM_DEFAULTS } from '@/lib/llm-defaults'
import {
  Settings,
  Save,
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Info,
  Cpu,
  Clock,
  Zap,
  Shield,
  Database,
  Bot,
  Wrench,
  Sparkles,
  Terminal,
  Coins
} from 'lucide-react'
import { InlineHelp } from '@/components/ui/help-tooltip'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { ToolLogo } from '@/components/ui/tool-logo'
import { Slider } from '@/components/ui/slider'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'react-hot-toast'

// API hooks
import { useAgent, useAgentConfig, useUpdateAgentConfig } from '@/hooks/use-agent-api'
import { useAgentModelConfig, useUpdateAgentModelConfig } from '@/hooks/use-model-api'
import { ModelSelector } from './model-selector'
import { Checkbox } from '@/components/ui/checkbox'
import { apiClient } from '@/lib/api-client'
import { useTools } from '@/hooks/use-tools-api'
import { useWorkspace } from '@/components/workspace-provider'

interface AgentConfigurationProps {
  agents: any[]
  selectedAgentId: string | null
  onAgentSelect: (agentId: string | null) => void
}

export function AgentConfiguration({
  agents,
  selectedAgentId,
  onAgentSelect
}: AgentConfigurationProps) {
  const { workspace } = useWorkspace()
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [configData, setConfigData] = useState<any>({})
  const [modelConfigData, setModelConfigData] = useState<any>({})
  const [assignedTools, setAssignedTools] = useState<number[]>([])

  // Plugin state
  const [workspacePlugins, setWorkspacePlugins] = useState<any[]>([])
  const [assignedPluginIds, setAssignedPluginIds] = useState<Set<string>>(new Set())
  const [pluginsLoading, setPluginsLoading] = useState(false)
  const [pluginsSaving, setPluginsSaving] = useState(false)

  // PRD-71: Skill assignment state
  const [workspaceSkills, setWorkspaceSkills] = useState<any[]>([])
  const [assignedSkillIds, setAssignedSkillIds] = useState<Set<number>>(new Set())
  const [skillsLoading, setSkillsLoading] = useState(false)
  const [skillsSaving, setSkillsSaving] = useState(false)

  // Fetch agent and configuration data
  const { data: agent, isLoading: agentLoading } = useAgent(selectedAgentId)
  const { data: agentConfig, isLoading: configLoading } = useAgentConfig(selectedAgentId)
  const updateConfigMutation = useUpdateAgentConfig()

  // PRD-15: Model configuration hooks
  const { data: agentModelConfig } = useAgentModelConfig(selectedAgentId ? Number(selectedAgentId) : null)
  const updateModelConfigMutation = useUpdateAgentModelConfig()

  // Tools API
  const { data: toolsData } = useTools({ status: 'active', limit: 100 })
  const availableTools: any[] = (toolsData as any)?.data || []

  // Initialize tools when agent loads
  useEffect(() => {
    if ((agent as any)?.tools && (agent as any).tools.length > 0) {
      const toolIds = (agent as any).tools.map((tool: any) => tool.id)
      setAssignedTools(toolIds)
    } else {
      setAssignedTools([])
    }
  }, [agent])

  // Fetch workspace-enabled plugins and agent plugin assignments
  useEffect(() => {
    if (!selectedAgentId) return
    const workspaceId = workspace?.id
    if (!workspaceId) return
    let mounted = true
    setPluginsLoading(true)
    ;(async () => {
      try {
        const [wpRes, apRes] = await Promise.all([
          apiClient.request<any>(`/api/workspaces/${workspaceId}/plugins`, { method: 'GET' }),
          apiClient.request<any>(`/api/agents/${selectedAgentId}/plugins`, { method: 'GET' }),
        ])
        if (!mounted) return
        const wpItems = wpRes?.items || wpRes || []
        const apItems = apRes?.items || apRes || []
        setWorkspacePlugins(Array.isArray(wpItems) ? wpItems : [])
        setAssignedPluginIds(new Set(
          (Array.isArray(apItems) ? apItems : []).map((p: any) => p.plugin_id)
        ))
      } catch (err) {
        console.error('Failed to fetch plugins:', err)
        if (mounted) {
          setWorkspacePlugins([])
          setAssignedPluginIds(new Set())
        }
      } finally {
        if (mounted) setPluginsLoading(false)
      }
    })()
    return () => { mounted = false }
  }, [selectedAgentId])

  // PRD-71: Fetch workspace-enabled skills and agent skill assignments
  useEffect(() => {
    if (!selectedAgentId) return
    const workspaceId = workspace?.id
    if (!workspaceId) return
    let mounted = true
    setSkillsLoading(true)
    ;(async () => {
      try {
        const [wsRes, asRes] = await Promise.all([
          apiClient.request<any>(`/api/workspaces/${workspaceId}/skills`, { method: 'GET' }),
          apiClient.request<any>(`/api/agents/${selectedAgentId}/skills`, { method: 'GET' }),
        ])
        if (!mounted) return
        const wsItems = wsRes?.items || []
        const agentSkills = asRes?.data || asRes || []
        setWorkspaceSkills(Array.isArray(wsItems) ? wsItems : [])
        setAssignedSkillIds(new Set(
          (Array.isArray(agentSkills) ? agentSkills : []).map((s: any) => s.id)
        ))
      } catch (err) {
        console.error('Failed to fetch skills:', err)
        if (mounted) {
          setWorkspaceSkills([])
          setAssignedSkillIds(new Set())
        }
      } finally {
        if (mounted) setSkillsLoading(false)
      }
    })()
    return () => { mounted = false }
  }, [selectedAgentId])

  // Initialize config data when agent config is loaded
  useEffect(() => {
    if (agentConfig) {
      const existingTags = Array.isArray((agent as any)?.tags)
        ? ((agent as any).tags as string[]).join(', ')
        : (agentConfig as any)?.tags || ''
      setConfigData({
        ...agentConfig,
        tags: existingTags
      })
      setHasUnsavedChanges(false)
    }
  }, [agentConfig, agent])

  // PRD-15: Initialize model config data
  useEffect(() => {
    if (agentModelConfig) {
      const modelConfig = (agentModelConfig as any)?.model_config || getDefaultModelConfig()
      setModelConfigData(modelConfig)
    }
  }, [agentModelConfig])

  // Handle form changes
  const handleConfigChange = (key: string, value: any) => {
    setConfigData((prev: any) => ({
      ...prev,
      [key]: value
    }))
    setHasUnsavedChanges(true)
  }

  // Handle nested config changes
  const handleNestedConfigChange = (section: string, key: string, value: any) => {
    setConfigData((prev: any) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }))
    setHasUnsavedChanges(true)
  }

  // PRD-15: Handle model config changes
  const handleModelConfigChange = (key: string, value: any) => {
    setModelConfigData((prev: any) => ({
      ...prev,
      [key]: value
    }))
    setHasUnsavedChanges(true)
  }

  // Handle plugin assignment toggle (persists immediately via API)
  const togglePluginAssignment = async (pluginId: string) => {
    if (!selectedAgentId) return
    const wasAssigned = assignedPluginIds.has(pluginId)
    const newIds = new Set(assignedPluginIds)
    if (wasAssigned) {
      newIds.delete(pluginId)
    } else {
      newIds.add(pluginId)
    }
    setAssignedPluginIds(newIds)
    setPluginsSaving(true)
    try {
      await apiClient.request(`/api/agents/${selectedAgentId}/plugins`, {
        method: 'PUT',
        body: { plugin_ids: Array.from(newIds) } as any,
      })
    } catch (err) {
      console.error('Failed to update plugin assignment:', err)
      setAssignedPluginIds(assignedPluginIds)
    } finally {
      setPluginsSaving(false)
    }
  }

  // Compute total token estimate for assigned plugins
  const assignedTokenEstimate = workspacePlugins
    .filter((p: any) => assignedPluginIds.has(p.plugin_id))
    .reduce((sum: number, p: any) => sum + (p.token_estimate || 0), 0)

  // PRD-71: Toggle skill assignment and persist via API
  const toggleSkillAssignment = async (skillId: number) => {
    if (!selectedAgentId) return
    const wasAssigned = assignedSkillIds.has(skillId)
    const newIds = new Set(assignedSkillIds)
    if (wasAssigned) {
      newIds.delete(skillId)
    } else {
      newIds.add(skillId)
    }

    setAssignedSkillIds(newIds)
    setSkillsSaving(true)

    try {
      if (wasAssigned) {
        await apiClient.request(`/api/agents/${selectedAgentId}/skills/${skillId}`, {
          method: 'DELETE',
        })
      } else {
        await apiClient.request(`/api/agents/${selectedAgentId}/skills`, {
          method: 'POST',
          body: JSON.stringify([skillId]),
        })
      }
    } catch (err) {
      console.error('Failed to update skill assignment:', err)
      setAssignedSkillIds(assignedSkillIds)
    } finally {
      setSkillsSaving(false)
    }
  }

  // PRD-71: Compute total token estimate for assigned skills
  const assignedSkillTokenEstimate = workspaceSkills
    .filter((s: any) => assignedSkillIds.has(s.skill_id))
    .reduce((sum: number, s: any) => sum + (s.estimated_tokens || 0), 0)

  // Handle tools assignment toggle
  const toggleToolAssignment = (toolId: number) => {
    setAssignedTools((prev) => {
      const newTools = prev.includes(toolId)
        ? prev.filter((id) => id !== toolId)
        : [...prev, toolId]
      setHasUnsavedChanges(true)
      return newTools
    })
  }

  // Save configuration
  const handleSave = async () => {
    if (!selectedAgentId) return

    try {
      toast.loading('Saving configuration...')
      const normalizedTags = Array.isArray(configData.tags)
        ? configData.tags
        : String(configData.tags || '')
          .split(',')
          .map((tag) => tag.trim())
          .filter((tag) => tag.length > 0)

      // 1. Update basic agent info (name, description)
      await updateConfigMutation.mutateAsync({
        agentId: selectedAgentId,
        config: {
          name: configData.name || (agent as any)?.name,
          description: configData.description || (agent as any)?.description,
          tags: normalizedTags,
          tool_ids: assignedTools
        }
      })

      // 2. Save model configuration (PRD-15)
      // Note: Plugin assignments are saved immediately via togglePluginAssignment
      let modelConfigFailed = false
      if (modelConfigData && Object.keys(modelConfigData).length > 0) {
        try {
          await updateModelConfigMutation.mutateAsync({
            agentId: Number(selectedAgentId),
            modelConfig: modelConfigData
          })
        } catch (error) {
          modelConfigFailed = true
          console.error('Failed to save model config:', error)
        }
      }

      toast.dismiss()
      if (modelConfigFailed) {
        toast.error('Agent saved, but model configuration failed to save. Please try again.')
        return
      }
      toast.success('Configuration saved!')
      setHasUnsavedChanges(false)

      // Force refresh agent data
      console.log('Reloading page to refresh agent data...')
      setTimeout(() => window.location.reload(), 500)
    } catch (error: any) {
      toast.dismiss()
      toast.error(error?.message || 'Failed to save configuration')
      console.error('Save configuration error:', error)
    }
  }

  // Reset configuration
  const handleReset = () => {
    if (agentConfig) {
      setConfigData(agentConfig)
      setHasUnsavedChanges(false)
    }
    if (agentModelConfig) {
      const modelConfig = (agentModelConfig as any)?.model_config || getDefaultModelConfig()
      setModelConfigData(modelConfig)
    }
    if (agent && (agent as any).tools && Array.isArray((agent as any).tools)) {
      const toolIds = (agent as any).tools.map((tool: any) => tool.id)
      setAssignedTools(toolIds)
    } else {
      setAssignedTools([])
    }
    setHasUnsavedChanges(false)
  }

  if (!selectedAgentId) {
    return (
      <Card className="glass-card">
        <CardContent className="p-12 text-center">
          <Settings className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Select an Agent</h3>
          <p className="text-muted-foreground">
            Choose an agent from the dropdown above to configure its settings
          </p>
          <div className="mt-6 max-w-md mx-auto">
            <Select
              value={selectedAgentId || ''}
              onValueChange={(value) => onAgentSelect(value)}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select an agent" />
              </SelectTrigger>
              <SelectContent>
                {agents.map((agent) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (agentLoading || configLoading) {
    return (
      <div className="space-y-6">
        <Card className="glass-card">
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-64" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-20 w-full" />
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Configuration</h2>
          <div className="flex items-center gap-2">
            <p className="text-muted-foreground">
              Configure settings and parameters for
            </p>
            <Select
              value={selectedAgentId || ''}
              onValueChange={(value) => onAgentSelect(value)}
            >
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Select an agent" />
              </SelectTrigger>
              <SelectContent>
                {agents.map((agent) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {hasUnsavedChanges && (
            <Badge variant="secondary" className="text-[hsl(var(--warning))]">
              <AlertTriangle className="w-3 h-3 mr-1" />
              Unsaved Changes
            </Badge>
          )}

          <Button
            variant="outline"
            onClick={handleReset}
            disabled={!hasUnsavedChanges || updateConfigMutation.isLoading}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>

          <Button
            variant="outline"
            onClick={handleSave}
            disabled={!hasUnsavedChanges || updateConfigMutation.isLoading}
          >
            <Save className="w-4 h-4 mr-2" />
            {updateConfigMutation.isLoading ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {/* Agent Info */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-primary-foreground">
              🤖
            </div>
            <div>
              <div>{(agent as any)?.name}</div>
              <p className="text-sm font-normal text-muted-foreground capitalize">
                {(agent as any)?.agent_type?.replace('_', ' ')} • {(agent as any)?.status}
              </p>
            </div>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Configuration Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Basic Configuration */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Basic Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="agent-name">Agent Name</Label>
              <Input
                id="agent-name"
                value={configData.name || (agent as any)?.name || ''}
                onChange={(e) => handleConfigChange('name', e.target.value)}
                placeholder="Enter agent name"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="agent-description">Description</Label>
              <Textarea
                id="agent-description"
                value={configData.description || (agent as any)?.description || ''}
                onChange={(e) => handleConfigChange('description', e.target.value)}
                placeholder="Describe the agent's purpose and capabilities"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="agent-tags">Tags (comma separated)</Label>
              <Input
                id="agent-tags"
                value={configData.tags || ''}
                onChange={(e) => handleConfigChange('tags', e.target.value)}
                placeholder="e.g. writing, pdf, research"
              />
              <p className="text-xs text-muted-foreground">
                Lightweight keywords that describe this agent&apos;s strengths.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="priority-level">Priority Level</Label>
              <Select
                value={configData.priority_level || (agent as any)?.priority_level || 'medium'}
                onValueChange={(value) => handleConfigChange('priority_level', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low Priority</SelectItem>
                  <SelectItem value="medium">Medium Priority</SelectItem>
                  <SelectItem value="high">High Priority</SelectItem>
                  <SelectItem value="critical">Critical Priority</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto Start</Label>
                <p className="text-sm text-muted-foreground">
                  Start agent automatically on system boot
                </p>
              </div>
              <Switch
                checked={configData.auto_start || (agent as any)?.auto_start || false}
                onCheckedChange={(checked) => handleConfigChange('auto_start', checked)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Performance Settings */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Performance Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Max Concurrent Tasks</Label>
              <div className="px-3">
                <Slider
                  value={[configData.max_concurrent_tasks || (agent as any)?.max_concurrent_tasks || 5]}
                  onValueChange={(value) => handleConfigChange('max_concurrent_tasks', value[0])}
                  max={20}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-muted-foreground mt-1">
                  <span>1</span>
                  <span>Current: {configData.max_concurrent_tasks || (agent as any)?.max_concurrent_tasks || 5}</span>
                  <span>20</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Task Timeout (seconds)</Label>
              <Input
                type="number"
                value={configData.task_timeout || '300'}
                onChange={(e) => handleConfigChange('task_timeout', parseInt(e.target.value) || 300)}
                min="10"
                max="3600"
              />
            </div>

            <div className="space-y-2">
              <Label>Retry Attempts</Label>
              <Input
                type="number"
                value={configData.retry_attempts || '3'}
                onChange={(e) => handleConfigChange('retry_attempts', parseInt(e.target.value) || 3)}
                min="0"
                max="10"
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable Caching</Label>
                <p className="text-sm text-muted-foreground">
                  Cache responses to improve performance
                </p>
              </div>
              <Switch
                checked={configData.enable_caching || false}
                onCheckedChange={(checked) => handleConfigChange('enable_caching', checked)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Security & Access
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Access Level</Label>
              <Select
                value={configData.access_level || 'standard'}
                onValueChange={(value) => handleConfigChange('access_level', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="restricted">Restricted</SelectItem>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="elevated">Elevated</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable Logging</Label>
                <p className="text-sm text-muted-foreground">
                  Log all agent activities
                </p>
              </div>
              <Switch
                checked={configData.enable_logging !== false}
                onCheckedChange={(checked) => handleConfigChange('enable_logging', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Rate Limiting</Label>
                <p className="text-sm text-muted-foreground">
                  Apply rate limits to API calls
                </p>
              </div>
              <Switch
                checked={configData.enable_rate_limiting || true}
                onCheckedChange={(checked) => handleConfigChange('enable_rate_limiting', checked)}
              />
            </div>

            <div className="space-y-2">
              <Label>API Rate Limit (calls/minute)</Label>
              <Input
                type="number"
                value={configData.api_rate_limit || '100'}
                onChange={(e) => handleConfigChange('api_rate_limit', parseInt(e.target.value) || 100)}
                min="1"
                max="1000"
              />
            </div>
          </CardContent>
        </Card>

        {/* Memory & Storage */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Memory & Storage
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Memory Limit (MB)</Label>
              <div className="px-3">
                <Slider
                  value={[configData.memory_limit || 512]}
                  onValueChange={(value) => handleConfigChange('memory_limit', value[0])}
                  max={4096}
                  min={128}
                  step={128}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-muted-foreground mt-1">
                  <span>128MB</span>
                  <span>Current: {configData.memory_limit || 512}MB</span>
                  <span>4GB</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Context Window Size</Label>
              <Select
                value={configData.context_window_size?.toString() || '8192'}
                onValueChange={(value) => handleConfigChange('context_window_size', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2048">2K Tokens</SelectItem>
                  <SelectItem value="4096">4K Tokens</SelectItem>
                  <SelectItem value="8192">8K Tokens</SelectItem>
                  <SelectItem value="16384">16K Tokens</SelectItem>
                  <SelectItem value="32768">32K Tokens</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Persistent Memory</Label>
                <p className="text-sm text-muted-foreground">
                  Retain memory between sessions
                </p>
              </div>
              <Switch
                checked={configData.persistent_memory || true}
                onCheckedChange={(checked) => handleConfigChange('persistent_memory', checked)}
              />
            </div>

            <div className="space-y-2">
              <Label>Memory Cleanup Interval (hours)</Label>
              <Input
                type="number"
                value={configData.memory_cleanup_interval || '24'}
                onChange={(e) => handleConfigChange('memory_cleanup_interval', parseInt(e.target.value) || 24)}
                min="1"
                max="168"
              />
            </div>
          </CardContent>
        </Card>

        {/* Plugin Assignment */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              Capability Assignment
            </CardTitle>
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Select marketplace capabilities to assign to this agent
              </p>
              {assignedPluginIds.size > 0 && (
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    <Coins className="w-3 h-3 mr-1" />
                    ~{assignedTokenEstimate.toLocaleString()} tokens
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    {assignedPluginIds.size} assigned
                  </Badge>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {pluginsLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              </div>
            ) : workspacePlugins.length > 0 ? (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {workspacePlugins.map((plugin: any) => {
                  const isAssigned = assignedPluginIds.has(plugin.plugin_id)
                  return (
                    <div
                      key={plugin.plugin_id}
                      className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors ${
                        isAssigned
                          ? 'bg-primary/5 border-primary/30'
                          : 'bg-background/50 border-border/50'
                      }`}
                    >
                      <Checkbox
                        id={`cfg-plugin-${plugin.plugin_id}`}
                        checked={isAssigned}
                        onCheckedChange={() => togglePluginAssignment(plugin.plugin_id)}
                        disabled={pluginsSaving}
                        className="mt-1"
                      />
                      <div className="flex-1 min-w-0">
                        <Label htmlFor={`cfg-plugin-${plugin.plugin_id}`} className="cursor-pointer">
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-medium truncate">{plugin.name}</span>
                            <Badge variant="outline" className="text-xs shrink-0">
                              v{plugin.version}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {plugin.description || 'No description available'}
                          </p>
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Terminal className="w-3 h-3" />
                              {plugin.skills_count} skills
                            </span>
                            <span className="flex items-center gap-1">
                              <Zap className="w-3 h-3" />
                              {plugin.commands_count} commands
                            </span>
                            <span className="flex items-center gap-1">
                              <Coins className="w-3 h-3" />
                              ~{(plugin.token_estimate || 0).toLocaleString()} tokens
                            </span>
                          </div>
                        </Label>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-8">
                <Sparkles className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Capabilities Available</h3>
                <p className="text-muted-foreground">
                  No capabilities are enabled for this workspace yet.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* PRD-71: Skill Assignment */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Skill Assignment
            </CardTitle>
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Assign individual skills to inject methodology into this agent
              </p>
              {assignedSkillIds.size > 0 && (
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    <Coins className="w-3 h-3 mr-1" />
                    ~{assignedSkillTokenEstimate.toLocaleString()} tokens
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    {assignedSkillIds.size} assigned
                  </Badge>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {skillsLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              </div>
            ) : workspaceSkills.length > 0 ? (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {workspaceSkills.map((skill: any) => {
                  const isAssigned = assignedSkillIds.has(skill.skill_id)
                  return (
                    <div
                      key={skill.skill_id}
                      className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors ${
                        isAssigned
                          ? 'bg-primary/5 border-primary/30'
                          : 'bg-background/50 border-border/50'
                      }`}
                    >
                      <Checkbox
                        id={`cfg-skill-${skill.skill_id}`}
                        checked={isAssigned}
                        onCheckedChange={() => toggleSkillAssignment(skill.skill_id)}
                        disabled={skillsSaving}
                        className="mt-1"
                      />
                      <div className="flex-1 min-w-0">
                        <Label htmlFor={`cfg-skill-${skill.skill_id}`} className="cursor-pointer">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="font-medium truncate">{skill.name}</span>
                              {skill.skill_version && (
                                <Badge variant="outline" className="text-xs shrink-0">
                                  v{skill.skill_version}
                                </Badge>
                              )}
                            </div>
                            {skill.category && (
                              <Badge variant="secondary" className="text-xs shrink-0">
                                {skill.category}
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {skill.description || 'No description available'}
                          </p>
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            {skill.estimated_tokens > 0 && (
                              <span className="flex items-center gap-1">
                                <Coins className="w-3 h-3" />
                                ~{skill.estimated_tokens.toLocaleString()} tokens
                              </span>
                            )}
                            {skill.skill_source && (
                              <span className="flex items-center gap-1">
                                <Terminal className="w-3 h-3" />
                                {skill.skill_source}
                              </span>
                            )}
                          </div>
                        </Label>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-6">
                <Zap className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">
                  No skills enabled for this workspace yet.
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Enable skills in Marketplace &gt; Capabilities &gt; Skills
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Tools Assignment */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wrench className="w-5 h-5 text-[hsl(var(--info))]" />
            Tool Assignment
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Select tools to grant this agent access to
          </p>
        </CardHeader>
        <CardContent>
          {availableTools && availableTools.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {availableTools.map((tool: any) => (
                <div key={tool.id} className="flex items-start space-x-3 p-3 bg-background/50 rounded-lg border border-border/50">
                  <Checkbox
                    id={`page-tool-${tool.id}`}
                    checked={assignedTools.includes(tool.id)}
                    onCheckedChange={() => toggleToolAssignment(tool.id)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <Label htmlFor={`page-tool-${tool.id}`} className="cursor-pointer">
                      <div className="flex flex-col space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium flex items-center gap-2">
                            <div className="flex items-center justify-center">
                              <ToolLogo
                                name={tool.name}
                                logo={tool.icon}
                                size={20}
                                showBackground={false}
                              />
                            </div>
                            {tool.name}
                          </span>
                          <Badge variant="outline" className="text-xs scale-90">
                            {tool.provider}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-2">
                          {tool.description || 'No description available'}
                        </p>
                      </div>
                    </Label>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Wrench className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Active Tools</h3>
              <p className="text-muted-foreground">
                No active tools available. Enable tools in Settings &gt; Tools first.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* PRD-15: Model Configuration */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-[hsl(var(--agent))]" />
            Model Configuration
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Select and configure the LLM model for this agent
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Selection */}
          <div className="space-y-3">
            <Label>AI Model</Label>
            <ModelSelector
              value={modelConfigData?.model_id || LLM_DEFAULTS.model_id}
              onChange={(modelId) => handleModelConfigChange('model_id', modelId)}
              agentType={(agent as any)?.agent_type}
            />
          </div>

          <Separator />

          {/* Temperature */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-1">Temperature <InlineHelp id="agents.config.model.temperature" size="sm" /></Label>
              <span className="text-sm text-muted-foreground">
                {modelConfigData?.temperature?.toFixed(2) || '0.70'}
              </span>
            </div>
            <Slider
              value={[modelConfigData?.temperature || 0.7]}
              onValueChange={(value) => handleModelConfigChange('temperature', value[0])}
              min={0}
              max={2}
              step={0.1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Controls randomness. Lower = more focused, Higher = more creative
            </p>
          </div>

          {/* Max Tokens */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-1">Max Output Tokens <InlineHelp id="agents.config.model.max_tokens" size="sm" /></Label>
              <span className="text-sm text-muted-foreground">
                {modelConfigData?.max_tokens || 2000}
              </span>
            </div>
            <Slider
              value={[modelConfigData?.max_tokens || 2000]}
              onValueChange={(value) => handleModelConfigChange('max_tokens', value[0])}
              min={100}
              max={16384}
              step={100}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Maximum tokens in the model's response
            </p>
          </div>

          {/* Advanced Model Settings */}
          <div className="space-y-4 pt-4 border-t border-border/50">
            <h4 className="text-sm font-medium">Advanced Model Settings</h4>

            {/* Top P */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-1">Top P (Nucleus Sampling) <InlineHelp id="agents.config.model.top_p" size="sm" /></Label>
                <span className="text-sm text-muted-foreground">
                  {modelConfigData?.top_p?.toFixed(2) || '1.00'}
                </span>
              </div>
              <Slider
                value={[modelConfigData?.top_p || 1.0]}
                onValueChange={(value) => handleModelConfigChange('top_p', value[0])}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
              />
            </div>

            {/* Frequency Penalty */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-1">Frequency Penalty <InlineHelp id="agents.config.model.frequency_penalty" size="sm" /></Label>
                <span className="text-sm text-muted-foreground">
                  {modelConfigData?.frequency_penalty?.toFixed(2) || '0.00'}
                </span>
              </div>
              <Slider
                value={[modelConfigData?.frequency_penalty || 0.0]}
                onValueChange={(value) => handleModelConfigChange('frequency_penalty', value[0])}
                min={0}
                max={2}
                step={0.1}
                className="w-full"
              />
            </div>

            {/* Presence Penalty */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-1">Presence Penalty <InlineHelp id="agents.config.model.presence_penalty" size="sm" /></Label>
                <span className="text-sm text-muted-foreground">
                  {modelConfigData?.presence_penalty?.toFixed(2) || '0.00'}
                </span>
              </div>
              <Slider
                value={[modelConfigData?.presence_penalty || 0.0]}
                onValueChange={(value) => handleModelConfigChange('presence_penalty', value[0])}
                min={0}
                max={2}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Configuration */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Advanced Configuration</CardTitle>
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              Advanced settings should only be modified by experienced users. Changes may affect agent performance.
            </AlertDescription>
          </Alert>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="custom-config">Custom Configuration (JSON)</Label>
            <Textarea
              id="custom-config"
              value={JSON.stringify(configData.custom_config || {}, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value)
                  handleConfigChange('custom_config', parsed)
                } catch (error) {
                  // Invalid JSON, don't update
                }
              }}
              placeholder='{"key": "value"}'
              rows={6}
              className="font-mono text-sm"
            />
          </div>
        </CardContent>
      </Card>

      {/* Save Status */}
      {hasUnsavedChanges && (
        <Alert className="border-[hsl(var(--warning))]/20 bg-[hsl(var(--warning))]/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            You have unsaved changes. Don't forget to save your configuration.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
