'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Save,
  Settings,
  Bot,
  AlertTriangle,
  CheckCircle,
  Info,
  Zap,
  Shield,
  Brain,
  Database,
  Network,
  Clock,
  Wrench,
  Puzzle,
  Terminal,
  Coins,
  ExternalLink,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { ToolLogo } from '@/components/ui/tool-logo'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { useAgent, useAgentConfig, useUpdateAgentConfig, useAgentSkills, useAddSkillToAgent, useRemoveSkillFromAgent } from '@/hooks/use-agent-api'
import { useSkillsApi } from '@/hooks/use-skills-api'
import { ModelSelector } from './model-selector'
import { useAgentModelConfig, useUpdateAgentModelConfig } from '@/hooks/use-model-api'
import { useTools } from '@/hooks/use-tools-api'
import { apiClient } from '@/lib/api-client'

interface AgentConfigurationModalProps {
  agentId: number | null
  open: boolean
  onClose: () => void
  onSave?: (agentId: number, config: any) => void
}

interface AgentConfiguration {
  id: number
  name: string
  description?: string
  agent_type: string
  status: string
  configuration?: {
    priority_level?: 'low' | 'medium' | 'high' | 'critical'
    max_concurrent_tasks?: number
    auto_start?: boolean
    retry_attempts?: number
    timeout_seconds?: number
    resource_limits?: {
      memory_mb?: number
      cpu_percent?: number
      network_bandwidth?: number
    }
    environment?: 'development' | 'staging' | 'production'
    logging_level?: 'debug' | 'info' | 'warning' | 'error'
    performance_monitoring?: boolean
  }
  available_skills?: Array<{
    id: number
    name: string
    description?: string
    skill_type: string
    category?: string
    is_active: boolean
    is_assigned?: boolean
  }>
}

import { AGENT_CATEGORIES, CATEGORY_TO_DB_MAP, DB_TO_CATEGORY_MAP } from '@/lib/agent-constants'

export function AgentConfigurationModal({
  agentId,
  open,
  onClose,
  onSave
}: AgentConfigurationModalProps) {
  const [activeTab, setActiveTab] = useState('general')
  const [hasChanges, setHasChanges] = useState(false)

  // Form state
  const [formData, setFormData] = useState<any>({})
  // Track the original DB agent_type so we can preserve it on save
  const [originalAgentType, setOriginalAgentType] = useState<string>('custom')

  // Use real API hooks
  const { data: agent, isLoading: loading, error: agentError } = useAgent(agentId?.toString() || '')
  const { data: agentConfig } = useAgentConfig(agentId?.toString() || '')
  // PRD-22: Load available skills from Skills API
  const { listSkills } = useSkillsApi()
  const [availableSkills, setAvailableSkills] = useState<any[]>([])
  const { data: agentSkills } = useAgentSkills(agentId?.toString() || '') // Get agent's current skills
  const updateConfigMutation = useUpdateAgentConfig()

  // PRD-15: Model configuration hooks
  const { data: agentModelConfig } = useAgentModelConfig(agentId)
  const updateModelConfigMutation = useUpdateAgentModelConfig()
  const addSkillMutation = useAddSkillToAgent()
  const removeSkillMutation = useRemoveSkillFromAgent()

  // Tools API
  const { data: toolsData } = useTools({ status: 'active', limit: 100 })
  const availableTools = toolsData?.data || []

  // PRD-42: Plugin assignment state
  const [workspacePlugins, setWorkspacePlugins] = useState<any[]>([])
  const [assignedPluginIds, setAssignedPluginIds] = useState<Set<string>>(new Set())
  const [pluginsLoading, setPluginsLoading] = useState(false)
  const [pluginsSaving, setPluginsSaving] = useState(false)

  const saving = updateConfigMutation.isLoading || updateModelConfigMutation.isLoading
  const error = (agentError as any)?.message || null

  // PRD-22: Fetch skills once when modal opens
  useEffect(() => {
    if (!open) return
    let mounted = true
      ; (async () => {
        try {
          const data = await listSkills({ limit: 200 })
          const items = Array.isArray(data) ? data : (data?.items || data?.skills || [])
          if (!mounted) return
          setAvailableSkills(items)
        } catch (_) {
          if (!mounted) return
          setAvailableSkills([])
        }
      })()
    return () => { mounted = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]) // Re-fetch when modal opens

  // PRD-42: Fetch workspace-enabled plugins and agent plugin assignments when modal opens
  useEffect(() => {
    if (!open || !agentId) return
    let mounted = true
    ;(async () => {
      setPluginsLoading(true)
      try {
        const workspaceId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
        if (!workspaceId) {
          if (mounted) setPluginsLoading(false)
          return
        }

        // Fetch workspace-enabled plugins and agent's assigned plugins in parallel
        const [wpRes, apRes] = await Promise.all([
          apiClient.request<any>(`/api/workspaces/${workspaceId}/plugins`, { method: 'GET' }),
          apiClient.request<any>(`/api/agents/${agentId}/plugins`, { method: 'GET' }),
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
  }, [open, agentId])

  useEffect(() => {
    if (agentConfig && agent && typeof agent === 'object') {
      // PRD-15: Get model config from agentModelConfig or use defaults
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

      // Initialize form data with real agent data
      const dbAgentType = (agent as any).agent_type || 'custom'
      const categoryName = DB_TO_CATEGORY_MAP[dbAgentType] || 'Custom'
      setOriginalAgentType(dbAgentType)

      setFormData({
        name: (agent as any).name || '',
        description: (agent as any).description || '',
        tags: Array.isArray((agent as any).tags) ? ((agent as any).tags as string[]).join(', ') : '',
        agent_type: categoryName,
        priority_level: (agentConfig as any).priority_level || 'medium',
        max_concurrent_tasks: (agentConfig as any).max_concurrent_tasks || 5,
        auto_start: (agentConfig as any).auto_start || false,
        retry_attempts: (agentConfig as any).retry_attempts || 3,
        timeout_seconds: (agentConfig as any).timeout_seconds || 300,
        memory_mb: (agentConfig as any).resource_limits?.memory_mb || 1024,
        cpu_percent: (agentConfig as any).resource_limits?.cpu_percent || 50,
        network_bandwidth: (agentConfig as any).resource_limits?.network_bandwidth || 100,
        environment: (agentConfig as any).environment || 'development',
        logging_level: (agentConfig as any).logging_level || 'info',
        performance_monitoring: (agentConfig as any).performance_monitoring || true,
        assigned_skills: (agent as any).skills?.map((skill: any) => skill.id) || [],
        // PRD-15: Model configuration
        model_config: modelConfig,
        // Tools configuration
        assigned_tools: (agent as any).tools?.map((tool: any) => tool.id) || []
      })
      setHasChanges(false)
    }
  }, [agentConfig, agent, agentModelConfig])


  const updateFormData = (key: string, value: any) => {
    setFormData((prev: any) => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  // PRD-15: Helper to update model config
  const updateModelConfig = (key: string, value: any) => {
    setFormData((prev: any) => ({
      ...prev,
      model_config: {
        ...prev.model_config,
        [key]: value
      }
    }))
    setHasChanges(true)
  }

  const toggleSkillAssignment = async (skillId: number) => {
    const currentSkills = formData.assigned_skills || []
    const hasSkill = currentSkills.includes(skillId)
    const newSkills = hasSkill
      ? currentSkills.filter((id: number) => id !== skillId)
      : [...currentSkills, skillId]

    // Optimistic UI update
    updateFormData('assigned_skills', newSkills)

    // Persist using the working backend endpoints
    try {
      if (hasSkill) {
        // DELETE: Use /api/v1/skills/agents/{agent_id}/skills with query params
        const deleteUrl = `/api/v1/skills/agents/${String(agentId)}/skills?skill_ids=${skillId}`
        const res = await fetch(deleteUrl, {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' }
        })
        console.info('Removed skill from agent', { agentId, skillId, ok: res.ok, status: res.status })
        if (!res.ok) {
          const errorText = await res.text()
          throw new Error(`Remove failed: ${res.status} - ${errorText}`)
        }
      } else {
        // POST: Use /api/agents/{agent_id}/skills - body must be raw array [1, 2, 3]
        const postUrl = `/api/agents/${String(agentId)}/skills`
        const res = await fetch(postUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify([skillId])
        })
        console.info('Added skill to agent', { agentId, skillId, ok: res.ok, status: res.status })
        if (!res.ok) {
          const errorText = await res.text()
          throw new Error(`Add failed: ${res.status} - ${errorText}`)
        }
      }
    } catch (e) {
      // Revert on error
      updateFormData('assigned_skills', currentSkills)
      console.error('Failed to update skill assignment', e)
    }
  }

  const toggleToolAssignment = (toolId: number) => {
    const currentTools = formData.assigned_tools || []
    const hasTool = currentTools.includes(toolId)
    const newTools = hasTool
      ? currentTools.filter((id: number) => id !== toolId)
      : [...currentTools, toolId]

    updateFormData('assigned_tools', newTools)
  }

  // PRD-42: Toggle plugin assignment and persist via API
  const togglePluginAssignment = async (pluginId: string) => {
    if (!agentId) return
    const wasAssigned = assignedPluginIds.has(pluginId)
    const newIds = new Set(assignedPluginIds)
    if (wasAssigned) {
      newIds.delete(pluginId)
    } else {
      newIds.add(pluginId)
    }

    // Optimistic update
    setAssignedPluginIds(newIds)
    setPluginsSaving(true)

    try {
      await apiClient.request(`/api/agents/${agentId}/plugins`, {
        method: 'PUT',
        body: { plugin_ids: Array.from(newIds) } as any,
      })
    } catch (err) {
      // Revert on error
      console.error('Failed to update plugin assignment:', err)
      setAssignedPluginIds(assignedPluginIds)
    } finally {
      setPluginsSaving(false)
    }
  }

  // PRD-42: Compute total token estimate for assigned plugins
  const assignedTokenEstimate = workspacePlugins
    .filter((p: any) => assignedPluginIds.has(p.plugin_id))
    .reduce((sum: number, p: any) => sum + (p.token_estimate || 0), 0)

  const handleSave = async () => {
    if (!agentId) {
      console.error('No agent ID provided')
      return
    }

    console.log('💾 Saving agent configuration...', {
      agentId,
      hasChanges,
      formData: formData
    })

    try {
      const tags = (formData.tags || '')
        .split(',')
        .map((tag: string) => tag.trim())
        .filter((tag: string) => tag.length > 0)

      // Convert category name to database agent_type value.
      // If the user's selected category maps to 'Custom' and the original DB type
      // was a specialized type (e.g. 'security_expert'), preserve the original.
      const selectedCategory = formData.agent_type || 'Custom'
      const mappedDbType = CATEGORY_TO_DB_MAP[selectedCategory] || 'custom'
      const originalMapsToCategory = DB_TO_CATEGORY_MAP[originalAgentType] || 'Custom'
      const dbAgentType =
        (mappedDbType === 'custom' && originalMapsToCategory === selectedCategory)
          ? originalAgentType
          : mappedDbType

      const updatePayload = {
        name: formData.name,
        description: formData.description,
        agent_type: dbAgentType,
        tags,
        configuration: {
          priority_level: formData.priority_level,
          max_concurrent_tasks: formData.max_concurrent_tasks,
          auto_start: formData.auto_start,
          retry_attempts: formData.retry_attempts,
          timeout_seconds: formData.timeout_seconds,
          resource_limits: {
            memory_mb: formData.memory_mb,
            cpu_percent: formData.cpu_percent,
            network_bandwidth: formData.network_bandwidth
          },
          environment: formData.environment,
          logging_level: formData.logging_level,
          performance_monitoring: formData.performance_monitoring,
          tags
        },
        skill_assignments: formData.assigned_skills,
        tool_ids: formData.assigned_tools
      }

      console.log('📝 Saving basic configuration...')
      // Save agent configuration
      await updateConfigMutation.mutateAsync({
        agentId: agentId.toString(),
        config: updatePayload
      })
      console.log('✅ Basic configuration saved')

      // PRD-15: Save model configuration
      if (formData.model_config) {
        console.log('🤖 Saving model configuration...', formData.model_config)
        await updateModelConfigMutation.mutateAsync({
          agentId: agentId,
          modelConfig: formData.model_config
        })
        console.log('✅ Model configuration saved')
      }

      if (onSave) {
        onSave(agentId, updatePayload)
      }

      setHasChanges(false)
      console.log('🎉 All changes saved successfully!')
      onClose()

    } catch (err) {
      console.error('❌ Error saving agent configuration:', err)
      // Show the full error to help debug
      alert(`Error saving configuration: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  if (!open || !agentId) return null

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      />
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-5xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3">
              <Settings className="w-6 h-6 text-orange-400" />
              <div>
                <span className="text-xl">Agent Configuration</span>
                <p className="text-sm text-muted-foreground font-normal">
                  {(agent as any)?.name || 'Loading...'}
                </p>
              </div>
            </CardTitle>
            <div className="flex items-center space-x-2">
              {hasChanges && (
                <Badge variant="secondary" className="text-xs">
                  Unsaved Changes
                </Badge>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={handleSave}
                disabled={saving || !hasChanges}
                className="hover:border-green-500/50"
              >
                {saving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Changes
                  </>
                )}
              </Button>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
          </CardHeader>

          <CardContent className="overflow-y-auto p-6">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading configuration...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error: {error}</p>
                  <Button onClick={() => window.location.reload()} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {agent && (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-6 bg-secondary/50">
                  <TabsTrigger value="general" className="flex items-center space-x-2">
                    <Info className="w-4 h-4" />
                    <span>General</span>
                  </TabsTrigger>
                  <TabsTrigger value="resources" className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>Resources</span>
                  </TabsTrigger>
                  <TabsTrigger value="skills" className="flex items-center space-x-2">
                    <Brain className="w-4 h-4" />
                    <span>Skills</span>
                  </TabsTrigger>
                  <TabsTrigger value="plugins" className="flex items-center space-x-2">
                    <Puzzle className="w-4 h-4" />
                    <span>Plugins</span>
                  </TabsTrigger>
                  <TabsTrigger value="model" className="flex items-center space-x-2">
                    <Bot className="w-4 h-4" />
                    <span>Model</span>
                  </TabsTrigger>
                  <TabsTrigger value="tools" className="flex items-center space-x-2">
                    <Wrench className="w-4 h-4" />
                    <span>Tools</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="general" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Basic Information</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="name">Agent Name</Label>
                        <Input
                          id="name"
                          value={formData.name || ''}
                          onChange={(e) => updateFormData('name', e.target.value)}
                          placeholder="Enter agent name"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="description">Description</Label>
                        <Textarea
                          id="description"
                          value={formData.description || ''}
                          onChange={(e) => updateFormData('description', e.target.value)}
                          placeholder="Enter agent description"
                          rows={3}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="tags">Tags (comma separated)</Label>
                        <Input
                          id="tags"
                          value={formData.tags || ''}
                          onChange={(e) => updateFormData('tags', e.target.value)}
                          placeholder="e.g. writing, pdf, research"
                        />
                        <p className="text-xs text-muted-foreground">
                          Lightweight keywords that describe the agent&apos;s strengths.
                        </p>
                      </div>

                      <div className="space-y-2">
                        <Label>Category</Label>
                        <Select
                          value={formData.agent_type || 'Custom'}
                          onValueChange={(value) => updateFormData('agent_type', value)}
                        >
                          <SelectTrigger>
                            {formData.agent_type ? (
                              <div className="flex items-center gap-2">
                                {(() => {
                                  const selected = AGENT_CATEGORIES.find(c => c.id === formData.agent_type)
                                  if (!selected) return <SelectValue />
                                  const Icon = selected.icon
                                  return (
                                    <>
                                      <Icon className={`w-4 h-4 ${selected.color}`} />
                                      <span>{selected.name}</span>
                                    </>
                                  )
                                })()}
                              </div>
                            ) : (
                              <SelectValue placeholder="Select category..." />
                            )}
                          </SelectTrigger>
                          <SelectContent>
                            {AGENT_CATEGORIES.map(cat => {
                              const Icon = cat.icon
                              return (
                                <SelectItem key={cat.id} value={cat.id}>
                                  <div className="flex items-center gap-2">
                                    <Icon className={`w-4 h-4 ${cat.color}`} />
                                    <span>{cat.name}</span>
                                  </div>
                                </SelectItem>
                              )
                            })}
                          </SelectContent>
                        </Select>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="resources" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Resource Limits</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="space-y-3">
                        <Label className="flex items-center space-x-2">
                          <Database className="w-4 h-4" />
                          <span>Memory Limit (MB)</span>
                        </Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.memory_mb || 1024]}
                            onValueChange={(value) => updateFormData('memory_mb', value[0])}
                            max={8192}
                            min={256}
                            step={256}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>256 MB</span>
                            <span className="font-medium">{formData.memory_mb || 1024} MB</span>
                            <span>8 GB</span>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label className="flex items-center space-x-2">
                          <Zap className="w-4 h-4" />
                          <span>CPU Limit (%)</span>
                        </Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.cpu_percent || 50]}
                            onValueChange={(value) => updateFormData('cpu_percent', value[0])}
                            max={100}
                            min={10}
                            step={5}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>10%</span>
                            <span className="font-medium">{formData.cpu_percent || 50}%</span>
                            <span>100%</span>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <Label className="flex items-center space-x-2">
                          <Network className="w-4 h-4" />
                          <span>Network Bandwidth (Mbps)</span>
                        </Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.network_bandwidth || 100]}
                            onValueChange={(value) => updateFormData('network_bandwidth', value[0])}
                            max={1000}
                            min={10}
                            step={10}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>10 Mbps</span>
                            <span className="font-medium">{formData.network_bandwidth || 100} Mbps</span>
                            <span>1000 Mbps</span>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label>Logging Level</Label>
                        <Select
                          value={formData.logging_level || 'info'}
                          onValueChange={(value) => updateFormData('logging_level', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="debug">Debug</SelectItem>
                            <SelectItem value="info">Info</SelectItem>
                            <SelectItem value="warning">Warning</SelectItem>
                            <SelectItem value="error">Error</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          Controls the verbosity of agent logs
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="skills" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Skills Assignment</CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Select skills to assign to this agent
                      </p>
                    </CardHeader>
                    <CardContent>
                      {availableSkills && availableSkills.length > 0 ? (
                        <div className="space-y-3">
                          {availableSkills.map((skill: any) => (
                            <div key={skill.id} className="flex items-center space-x-3 p-3 bg-background/50 rounded-lg">
                              <Checkbox
                                id={`skill-${skill.id}`}
                                checked={formData.assigned_skills?.includes(skill.id) || false}
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
                          <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Skills Available</h3>
                          <p className="text-muted-foreground">
                            No skills are available for assignment at this time.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* PRD-42: Plugins Tab */}
                <TabsContent value="plugins" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Puzzle className="h-5 w-5 text-orange-400" />
                        Plugin Assignment
                      </CardTitle>
                      <div className="flex items-center justify-between">
                        <p className="text-sm text-muted-foreground">
                          Select marketplace plugins to assign to this agent
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
                        <div className="space-y-3">
                          {workspacePlugins.map((plugin: any) => {
                            const isAssigned = assignedPluginIds.has(plugin.plugin_id)
                            return (
                              <div
                                key={plugin.plugin_id}
                                className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors ${
                                  isAssigned
                                    ? 'bg-orange-500/5 border-orange-500/30'
                                    : 'bg-background/50 border-border/50'
                                }`}
                              >
                                <Checkbox
                                  id={`plugin-${plugin.plugin_id}`}
                                  checked={isAssigned}
                                  onCheckedChange={() => togglePluginAssignment(plugin.plugin_id)}
                                  disabled={pluginsSaving}
                                  className="mt-1"
                                />
                                <div className="flex-1 min-w-0">
                                  <Label htmlFor={`plugin-${plugin.plugin_id}`} className="cursor-pointer">
                                    <div className="flex items-center justify-between gap-2">
                                      <div className="flex items-center gap-2 min-w-0">
                                        <span className="font-medium truncate">{plugin.name}</span>
                                        <Badge variant="outline" className="text-xs shrink-0">
                                          v{plugin.version}
                                        </Badge>
                                      </div>
                                      <div className="flex items-center gap-2 shrink-0">
                                        {plugin.security_status === 'safe' && (
                                          <Badge variant="secondary" className="text-xs text-green-400 border-green-500/30">
                                            <Shield className="w-3 h-3 mr-1" />
                                            Verified
                                          </Badge>
                                        )}
                                        {plugin.category_name && (
                                          <Badge variant="secondary" className="text-xs">
                                            {plugin.category_name}
                                          </Badge>
                                        )}
                                      </div>
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
                          <Puzzle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Plugins Available</h3>
                          <p className="text-muted-foreground text-sm mb-4">
                            No plugins are enabled for this workspace yet.
                          </p>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              window.location.href = '/marketplace'
                            }}
                          >
                            <ExternalLink className="w-4 h-4 mr-2" />
                            Browse Marketplace
                          </Button>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="tools" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Wrench className="h-5 w-5 text-blue-400" />
                        Tool Assignment
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Select tools to grant this agent access to
                      </p>
                    </CardHeader>
                    <CardContent>
                      {availableTools && availableTools.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {availableTools.map((tool: any) => (
                            <div key={tool.id} className="flex items-start space-x-3 p-3 bg-background/50 rounded-lg border border-border/50">
                              <Checkbox
                                id={`tool-${tool.id}`}
                                checked={formData.assigned_tools?.includes(tool.id) || false}
                                onCheckedChange={() => toggleToolAssignment(tool.id)}
                                className="mt-1"
                              />
                              <div className="flex-1">
                                <Label htmlFor={`tool-${tool.id}`} className="cursor-pointer">
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
                            No active tools available. Enable tools in Settings {'->'} Tools first.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* PRD-15: Model Configuration Tab */}
                <TabsContent value="model" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Bot className="h-5 w-5 text-purple-400" />
                        Model Configuration
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Select and configure the LLM model for this agent
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Model Selection */}
                      <ModelSelector
                        value={formData.model_config?.model_id || 'gpt-4'}
                        onChange={(modelId) => updateModelConfig('model_id', modelId)}
                        agentType={formData.agent_type}
                      />

                      <Separator />

                      {/* Temperature */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="temperature">Temperature</Label>
                          <span className="text-sm text-muted-foreground">
                            {formData.model_config?.temperature?.toFixed(2) || '0.70'}
                          </span>
                        </div>
                        <Slider
                          id="temperature"
                          value={[formData.model_config?.temperature || 0.7]}
                          onValueChange={([value]) => updateModelConfig('temperature', value)}
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
                          <Label htmlFor="max-tokens">Max Output Tokens</Label>
                          <span className="text-sm text-muted-foreground">
                            {formData.model_config?.max_tokens || 2000}
                          </span>
                        </div>
                        <Slider
                          id="max-tokens"
                          value={[formData.model_config?.max_tokens || 2000]}
                          onValueChange={([value]) => updateModelConfig('max_tokens', value)}
                          min={100}
                          max={4000}
                          step={100}
                          className="w-full"
                        />
                        <p className="text-xs text-muted-foreground">
                          Maximum tokens in the model's response
                        </p>
                      </div>

                      {/* Advanced Settings */}
                      <div className="space-y-4 pt-4 border-t border-border/50">
                        <h4 className="text-sm font-medium text-foreground">Advanced Settings</h4>

                        {/* Top P */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Label htmlFor="top-p" className="text-xs">Top P (Nucleus Sampling)</Label>
                            <span className="text-xs text-muted-foreground">
                              {formData.model_config?.top_p?.toFixed(2) || '1.00'}
                            </span>
                          </div>
                          <Slider
                            id="top-p"
                            value={[formData.model_config?.top_p || 1.0]}
                            onValueChange={([value]) => updateModelConfig('top_p', value)}
                            min={0}
                            max={1}
                            step={0.05}
                            className="w-full"
                          />
                        </div>

                        {/* Frequency Penalty */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Label htmlFor="frequency-penalty" className="text-xs">Frequency Penalty</Label>
                            <span className="text-xs text-muted-foreground">
                              {formData.model_config?.frequency_penalty?.toFixed(2) || '0.00'}
                            </span>
                          </div>
                          <Slider
                            id="frequency-penalty"
                            value={[formData.model_config?.frequency_penalty || 0.0]}
                            onValueChange={([value]) => updateModelConfig('frequency_penalty', value)}
                            min={0}
                            max={2}
                            step={0.1}
                            className="w-full"
                          />
                        </div>

                        {/* Presence Penalty */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Label htmlFor="presence-penalty" className="text-xs">Presence Penalty</Label>
                            <span className="text-xs text-muted-foreground">
                              {formData.model_config?.presence_penalty?.toFixed(2) || '0.00'}
                            </span>
                          </div>
                          <Slider
                            id="presence-penalty"
                            value={[formData.model_config?.presence_penalty || 0.0]}
                            onValueChange={([value]) => updateModelConfig('presence_penalty', value)}
                            min={0}
                            max={2}
                            step={0.1}
                            className="w-full"
                          />
                        </div>
                      </div>

                      {/* Fallback Model */}
                      <div className="space-y-2">
                        <Label htmlFor="fallback-model">Fallback Model (Optional)</Label>
                        <Select
                          value={formData.model_config?.fallback_model_id || 'none'}
                          onValueChange={(value) => updateModelConfig('fallback_model_id', value === 'none' ? null : value)}
                        >
                          <SelectTrigger id="fallback-model" className="bg-background/50 border-border">
                            <SelectValue placeholder="Select fallback model..." />
                          </SelectTrigger>
                          <SelectContent className="bg-gray-900 border-gray-800">
                            <SelectItem value="none">No fallback</SelectItem>
                            <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                            <SelectItem value="claude-3-haiku-20240307">Claude 3 Haiku</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                          Model to use if primary model fails or is unavailable
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
