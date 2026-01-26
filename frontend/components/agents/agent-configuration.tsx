
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
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
  Wrench
} from 'lucide-react'
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
import { useAgent, useAgentConfig, useUpdateAgentConfig, useAgentSkills, useSkills } from '@/hooks/use-agent-api'
import { useAgentModelConfig, useUpdateAgentModelConfig } from '@/hooks/use-model-api'
import { ModelSelector } from './model-selector'
import { Checkbox } from '@/components/ui/checkbox'
import { apiClient } from '@/lib/api-client'
import { useTools } from '@/hooks/use-tools-api'

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
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [configData, setConfigData] = useState<any>({})
  const [modelConfigData, setModelConfigData] = useState<any>({})
  const [assignedSkills, setAssignedSkills] = useState<number[]>([])
  const [assignedTools, setAssignedTools] = useState<number[]>([])

  // Fetch agent and configuration data
  const { data: agent, isLoading: agentLoading } = useAgent(selectedAgentId)
  const { data: agentConfig, isLoading: configLoading } = useAgentConfig(selectedAgentId)
  const { data: availableSkills } = useSkills() // Get ALL available skills
  const { data: agentSkills } = useAgentSkills(selectedAgentId) // Get agent's current skills
  const updateConfigMutation = useUpdateAgentConfig()

  // PRD-15: Model configuration hooks
  const { data: agentModelConfig } = useAgentModelConfig(selectedAgentId ? Number(selectedAgentId) : null)
  const updateModelConfigMutation = useUpdateAgentModelConfig()

  // Tools API
  const { data: toolsData } = useTools({ status: 'active', limit: 100 })
  const availableTools: any[] = (toolsData as any)?.data || []

  // Initialize assigned skills when agent or agentSkills loads
  useEffect(() => {
    if (agentSkills && agentSkills.length > 0) {
      const skillIds = agentSkills.map((skill: any) => skill.id)
      console.log('Initializing assigned skills:', skillIds)
      setAssignedSkills(skillIds)
    } else if ((agent as any)?.skills && (agent as any).skills.length > 0) {
      const skillIds = (agent as any).skills.map((skill: any) => skill.id)
      console.log('Initializing assigned skills from agent:', skillIds)
      setAssignedSkills(skillIds)
    } else {
      setAssignedSkills([])
    }

    // Initialize tools
    if ((agent as any)?.tools && (agent as any).tools.length > 0) {
      const toolIds = (agent as any).tools.map((tool: any) => tool.id)
      setAssignedTools(toolIds)
    } else {
      setAssignedTools([])
    }
  }, [agentSkills, agent])

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
      const modelConfig = (agentModelConfig as any)?.model_config || {
        provider: 'openai',
        model_id: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        fallback_model_id: null
      }
      setModelConfigData(modelConfig)
    }
  }, [agentModelConfig])

  // Initialize assigned skills when agent data loads
  useEffect(() => {
    if (agentSkills && Array.isArray(agentSkills)) {
      const skillIds = agentSkills.map((skill: any) => skill.id)
      setAssignedSkills(skillIds)
    } else if (agent && (agent as any).skills && Array.isArray((agent as any).skills)) {
      const skillIds = (agent as any).skills.map((skill: any) => skill.id)
      setAssignedSkills(skillIds)
    }
  }, [agentSkills, agent])

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

  // Handle skills assignment toggle
  const toggleSkillAssignment = (skillId: number) => {
    setAssignedSkills((prev) => {
      const newSkills = prev.includes(skillId)
        ? prev.filter((id) => id !== skillId)
        : [...prev, skillId]
      setHasUnsavedChanges(true)
      return newSkills
    })
  }

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

      // 2. Handle skill assignments separately
      // Get current skills
      const currentSkillIds = (agent as any)?.skills?.map((s: any) => s.id) || []
      const newSkillIds = assignedSkills

      console.log('Skill assignment - Current:', currentSkillIds)
      console.log('Skill assignment - New:', newSkillIds)

      // Find skills to add and remove
      const skillsToAdd = newSkillIds.filter((id: number) => !currentSkillIds.includes(id))
      const skillsToRemove = currentSkillIds.filter((id: number) => !newSkillIds.includes(id))

      console.log('Skills to add:', skillsToAdd)
      console.log('Skills to remove:', skillsToRemove)

      // Add new skills
      for (const skillId of skillsToAdd) {
        try {
          console.log(`Adding skill ${skillId} to agent ${selectedAgentId}`)
          const result = await apiClient.addSkillToAgent(selectedAgentId, String(skillId))
          console.log(`Successfully added skill ${skillId}:`, result)
        } catch (error) {
          console.error(`Failed to add skill ${skillId}:`, error)
          toast.error(`Failed to add skill ${skillId}`)
        }
      }

      // Remove old skills
      for (const skillId of skillsToRemove) {
        try {
          console.log(`Removing skill ${skillId} from agent ${selectedAgentId}`)
          const result = await apiClient.removeSkillFromAgent(selectedAgentId, String(skillId))
          console.log(`Successfully removed skill ${skillId}:`, result)
        } catch (error) {
          console.error(`Failed to remove skill ${skillId}:`, error)
          toast.error(`Failed to remove skill ${skillId}`)
        }
      }

      // 3. Save model configuration (PRD-15)
      if (modelConfigData && Object.keys(modelConfigData).length > 0) {
        try {
          await updateModelConfigMutation.mutateAsync({
            agentId: Number(selectedAgentId),
            modelConfig: modelConfigData
          })
        } catch (error) {
          console.error('Failed to save model config:', error)
          toast.error('Some settings saved, but model configuration failed')
        }
      }

      toast.dismiss()
      toast.success(`Configuration saved! Added ${skillsToAdd.length} skills, removed ${skillsToRemove.length} skills.`)
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
      const modelConfig = (agentModelConfig as any)?.model_config || {
        provider: 'openai',
        model_id: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        fallback_model_id: null
      }
      setModelConfigData(modelConfig)
    }
    // Reset skills to original state
    if (agentSkills && Array.isArray(agentSkills)) {
      const skillIds = agentSkills.map((skill: any) => skill.id)
      setAssignedSkills(skillIds)
    } else if (agent && (agent as any).skills && Array.isArray((agent as any).skills)) {
      const skillIds = (agent as any).skills.map((skill: any) => skill.id)
      setAssignedSkills(skillIds)
    } else {
      setAssignedSkills([])
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
            <Badge variant="secondary" className="text-yellow-600">
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
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-white">
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

        {/* Skills Assignment */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="w-5 h-5" />
              Skills Assignment
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Select skills to assign to this agent
            </p>
          </CardHeader>
          <CardContent>
            {availableSkills && availableSkills.length > 0 ? (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {availableSkills.map((skill: any) => (
                  <div key={skill.id} className="flex items-center space-x-3 p-3 bg-background/50 rounded-lg">
                    <Checkbox
                      id={`skill-${skill.id}`}
                      checked={assignedSkills.includes(skill.id)}
                      onCheckedChange={() => toggleSkillAssignment(skill.id)}
                    />
                    <div className="flex-1">
                      <Label htmlFor={`skill-${skill.id}`} className="cursor-pointer">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-medium">{skill.name}</h4>
                            <p className="text-sm text-muted-foreground">
                              {skill.description || 'No description available'}
                            </p>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline" className="text-xs">
                              {skill.skill_type?.replace('_', ' ') || 'Unknown'}
                            </Badge>
                            {skill.category && (
                              <Badge variant="secondary" className="text-xs">
                                {skill.category}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </Label>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Bot className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Skills Available</h3>
                <p className="text-muted-foreground">
                  No skills are available for assignment at this time.
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
            <Wrench className="w-5 h-5 text-blue-400" />
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
            <Bot className="w-5 h-5 text-purple-400" />
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
              value={modelConfigData?.model_id || 'gpt-4'}
              onChange={(modelId) => handleModelConfigChange('model_id', modelId)}
              agentType={(agent as any)?.agent_type}
            />
          </div>

          <Separator />

          {/* Temperature */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Temperature</Label>
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
              <Label>Max Output Tokens</Label>
              <span className="text-sm text-muted-foreground">
                {modelConfigData?.max_tokens || 2000}
              </span>
            </div>
            <Slider
              value={[modelConfigData?.max_tokens || 2000]}
              onValueChange={(value) => handleModelConfigChange('max_tokens', value[0])}
              min={100}
              max={4000}
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
                <Label>Top P (Nucleus Sampling)</Label>
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
                <Label>Frequency Penalty</Label>
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
                <Label>Presence Penalty</Label>
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
        <Alert className="border-yellow-200 bg-yellow-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            You have unsaved changes. Don't forget to save your configuration.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
