'use client'

import * as React from 'react'
import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getDefaultModelConfig, LLM_DEFAULTS } from '@/lib/llm-defaults'
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
  Database,
  Network,
  Clock,
  Wrench,
  Sparkles,
  Terminal,
  Coins,
  ExternalLink,
  User,
  PenLine,
  ChevronDown,
  ChevronUp,
  Activity,
  Loader2,
  Play,
} from 'lucide-react'
import { InlineHelp } from '@/components/ui/help-tooltip'
import { Button } from '@/components/ui/button'
import { ErrorState, LoadingState } from '@/components/shared'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { ToolLogo } from '@/components/ui/tool-logo'
import { PremiumIcon } from '@/components/shared'
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
import { useWorkspace } from '@/components/workspace-provider'
import { useSkillsApi } from '@/hooks/use-skills-api'
import { ModelSelector } from './model-selector'
import { useAgentModelConfig, useUpdateAgentModelConfig } from '@/hooks/use-model-api'
import { useTools } from '@/hooks/use-tools-api'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

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
}

import { AGENT_CATEGORIES, CATEGORY_TO_DB_MAP, DB_TO_CATEGORY_MAP } from '@/lib/agent-constants'

export function AgentConfigurationModal({
  agentId,
  open,
  onClose,
  onSave
}: AgentConfigurationModalProps) {
  const { workspace } = useWorkspace()
  const [activeTab, setActiveTab] = useState('general')
  const [hasChanges, setHasChanges] = useState(false)

  // Form state
  const [formData, setFormData] = useState<any>({})
  // Track the original DB agent_type so we can preserve it on save
  const [originalAgentType, setOriginalAgentType] = useState<string>('custom')

  // Use real API hooks
  const { data: agent, isLoading: loading, error: agentError, refetch: refetchAgent } = useAgent(agentId?.toString() || '')
  const { data: agentConfig } = useAgentConfig(agentId?.toString() || '')
  const updateConfigMutation = useUpdateAgentConfig()

  // Get system icons
  const { data: iconMappings = {} } = useSystemIcons()

  // PRD-15: Model configuration hooks
  const { data: agentModelConfig } = useAgentModelConfig(agentId)
  const updateModelConfigMutation = useUpdateAgentModelConfig()
  // Tools API
  const { data: toolsData } = useTools({ status: 'active', limit: 100 })
  const availableTools = toolsData?.data || []

  // PRD-42: Plugin assignment state
  const [workspacePlugins, setWorkspacePlugins] = useState<any[]>([])
  const [assignedPluginIds, setAssignedPluginIds] = useState<Set<string>>(new Set())
  const [pluginsLoading, setPluginsLoading] = useState(false)
  const [pluginsSaving, setPluginsSaving] = useState(false)

  // PRD-71: Skill assignment state
  const [workspaceSkills, setWorkspaceSkills] = useState<any[]>([])
  const [assignedSkillIds, setAssignedSkillIds] = useState<Set<number>>(new Set())
  const [skillsLoading, setSkillsLoading] = useState(false)
  const [skillsSaving, setSkillsSaving] = useState(false)

  // US-023: Persona state
  type PersonaMode = 'none' | 'predefined' | 'custom'
  const [personaMode, setPersonaMode] = useState<PersonaMode>('none')
  const [selectedPersonaId, setSelectedPersonaId] = useState<string | null>(null)
  const [currentPersonaName, setCurrentPersonaName] = useState<string | null>(null)
  const [currentPersonaPrompt, setCurrentPersonaPrompt] = useState<string | null>(null)
  const [customPersonaPrompt, setCustomPersonaPrompt] = useState('')
  const [personas, setPersonas] = useState<any[]>([])
  const [personasLoading, setPersonasLoading] = useState(false)
  const [personaCategoryFilter, setPersonaCategoryFilter] = useState<string>('all')
  const [expandedPersonaId, setExpandedPersonaId] = useState<string | null>(null)
  const [personaSaving, setPersonaSaving] = useState(false)
  const [personaLoaded, setPersonaLoaded] = useState(false)

  // PRD-55: Heartbeat configuration state
  const [heartbeatConfig, setHeartbeatConfig] = useState({
    enabled: false,
    interval_minutes: 60,
    inherit_active_hours: true,
    active_hours_start: '08:00',
    active_hours_end: '20:00',
    prompt: '',
    auto_act: false,
    report_to: 'orchestrator',
    webhook_url: '',
    channel_id: '',
  })
  const [heartbeatRunning, setHeartbeatRunning] = useState(false)
  const [lastHeartbeatResult, setLastHeartbeatResult] = useState<any>(null)
  const [connectedIntegrations, setConnectedIntegrations] = useState<Array<{ key: string; platform: string }>>([])


  const saving = updateConfigMutation.isLoading || updateModelConfigMutation.isLoading
  const error = (agentError as any)?.message || null

  // Track whether form has been initialized for this modal session.
  // Prevents polling refetches from overwriting user's in-progress edits.
  const formInitializedRef = useRef(false)

  // Reset when modal closes
  useEffect(() => {
    if (!open) {
      formInitializedRef.current = false
    }
  }, [open])

  // Reset when agentId changes (even while modal is open) so the new agent's data initializes
  useEffect(() => {
    formInitializedRef.current = false
  }, [agentId])

  // PRD-42: Fetch workspace-enabled plugins and agent plugin assignments when modal opens
  useEffect(() => {
    if (!open || !agentId) return
    const workspaceId = workspace?.id
    if (!workspaceId) return
    let mounted = true
      ; (async () => {
        setPluginsLoading(true)
        try {
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
  }, [open, agentId, workspace?.id])

  // PRD-71: Fetch workspace-enabled skills and agent skill assignments when modal opens
  useEffect(() => {
    if (!open || !agentId) return
    // Wait for workspace context to load — reading localStorage directly was racy
    // for new users whose last_active_workspace wasn't persisted yet.
    const workspaceId = workspace?.id
    if (!workspaceId) return
    let mounted = true
      ; (async () => {
        setSkillsLoading(true)
        try {
          const [wsRes, asRes] = await Promise.all([
            apiClient.request<any>(`/api/workspaces/${workspaceId}/skills`, { method: 'GET' }),
            apiClient.request<any>(`/api/agents/${agentId}/skills`, { method: 'GET' }),
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
  }, [open, agentId, workspace?.id])

  // US-023: Fetch personas list and current agent persona when modal opens
  useEffect(() => {
    if (!open || !agentId) return
    let mounted = true

    // Fetch personas list
    setPersonasLoading(true)
    apiClient.request<any>('/api/personas')
      .then((data) => {
        if (!mounted) return
        const list = Array.isArray(data) ? data : (data?.items || data?.personas || data?.data || [])
        setPersonas(list)
      })
      .catch((err) => {
        console.error('Failed to fetch personas:', err)
        if (mounted) setPersonas([])
      })
      .finally(() => { if (mounted) setPersonasLoading(false) })

    // Fetch current agent persona
    apiClient.request<any>(`/api/agents/${agentId}/persona`)
      .then((data) => {
        if (!mounted) return
        setPersonaLoaded(true)
        if (data?.use_custom_persona && data?.custom_persona_prompt) {
          setPersonaMode('custom')
          setCustomPersonaPrompt(data.custom_persona_prompt)
          setCurrentPersonaName(null)
          setCurrentPersonaPrompt(data.custom_persona_prompt)
          setSelectedPersonaId(null)
        } else if (data?.persona_id) {
          setPersonaMode('predefined')
          setSelectedPersonaId(data.persona_id)
          setCurrentPersonaName(data.persona_name || data.name || null)
          setCurrentPersonaPrompt(data.system_prompt || null)
          setCustomPersonaPrompt('')
        } else {
          setPersonaMode('none')
          setSelectedPersonaId(null)
          setCurrentPersonaName(null)
          setCurrentPersonaPrompt(null)
          setCustomPersonaPrompt('')
        }
      })
      .catch((err) => {
        console.error('Failed to fetch agent persona:', err)
        if (mounted) {
          setPersonaLoaded(true)
          setPersonaMode('none')
        }
      })

    return () => { mounted = false }
  }, [open, agentId])

  // PRD-55: Load heartbeat config + connected channels when modal opens
  useEffect(() => {
    if (!open || !agentId) return
    let mounted = true
    apiClient.request<any>(`/api/heartbeat/agents/${agentId}/config`)
      .then((data) => {
        if (!mounted) return
        if (data) {
          setHeartbeatConfig(prev => ({ ...prev, ...data }))
        }
      })
      .catch(() => { })
    // Load last heartbeat result
    apiClient.request<any>(`/api/heartbeat/agents/${agentId}/last`)
      .then((data) => {
        if (!mounted) return
        if (data) setLastHeartbeatResult(data)
      })
      .catch(() => { })
    // Load connected messaging platforms for Report To dropdown
    // Check both workspace integrations AND channel_connections
    Promise.all([
      apiClient.request<any>(`/api/workspaces/current/integrations`).catch(() => ({})),
      apiClient.request<any>(`/api/channels`).catch(() => []),
    ]).then(([integrations, channels]) => {
      if (!mounted) return
      const found: Array<{ key: string; platform: string }> = []
      const seen = new Set<string>()
      // From workspace integrations
      const platformMap: Record<string, string> = { telegram_bot_token: 'telegram', slack_bot_token: 'slack' }
      if (integrations) {
        for (const [key, val] of Object.entries(integrations)) {
          const platform = platformMap[key]
          if (platform && (val as any)?.configured && !seen.has(platform)) {
            found.push({ key, platform })
            seen.add(platform)
          }
        }
      }
      // From channel_connections
      if (Array.isArray(channels)) {
        for (const ch of channels) {
          if (!seen.has(ch.platform)) {
            found.push({ key: `channel:${ch.id}`, platform: ch.platform })
            seen.add(ch.platform)
          }
        }
      }
      setConnectedIntegrations(found)
    })
    return () => { mounted = false }
  }, [open, agentId])

  // US-023: Pre-fill custom prompt when switching from predefined to custom
  useEffect(() => {
    if (personaMode === 'custom' && selectedPersonaId && !customPersonaPrompt) {
      const persona = personas.find((p: any) => p.id === selectedPersonaId)
      if (persona?.system_prompt) {
        setCustomPersonaPrompt(persona.system_prompt)
      }
    }
  }, [personaMode, selectedPersonaId, personas, customPersonaPrompt])

  useEffect(() => {
    // Only initialize form data once per modal session.
    // useAgent polls every 10s — without this guard, refetches overwrite
    // the user's in-progress edits and reset hasChanges to false.
    if (formInitializedRef.current) return
    if (agentConfig && agent && typeof agent === 'object') {
      formInitializedRef.current = true

      // PRD-15: Get model config from agentModelConfig or use defaults
      const modelConfig = (agentModelConfig as any)?.model_config || getDefaultModelConfig()

      // Initialize form data with real agent data
      const dbAgentType = (agent as any).agent_type || 'custom'
      // Prefer marketplace_category (the actual UI category), fall back to configuration.category, then DB mapping
      const categoryName = (agent as any).marketplace_category
        || (agent as any).configuration?.category
        || DB_TO_CATEGORY_MAP[dbAgentType]
        || 'custom'
      setOriginalAgentType(dbAgentType)

      setFormData({
        name: (agent as any).name || '',
        description: (agent as any).description || '',
        job_title: (agent as any).job_title || '',
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
        // Tools configuration — filter out null IDs (tools without cache entries)
        assigned_tools: ((agent as any).tools || []).map((tool: any) => tool.id).filter((id: any) => id != null)
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

  // PRD-71: Toggle skill assignment and persist via API
  const toggleSkillAssignment = async (skillId: number) => {
    if (!agentId) return
    const wasAssigned = assignedSkillIds.has(skillId)
    const newIds = new Set(assignedSkillIds)
    if (wasAssigned) {
      newIds.delete(skillId)
    } else {
      newIds.add(skillId)
    }

    // Optimistic update
    setAssignedSkillIds(newIds)
    setSkillsSaving(true)

    try {
      if (wasAssigned) {
        await apiClient.request(`/api/agents/${agentId}/skills/${skillId}`, {
          method: 'DELETE',
        })
      } else {
        await apiClient.request(`/api/agents/${agentId}/skills`, {
          method: 'POST',
          body: JSON.stringify([skillId]),
        })
      }
    } catch (err) {
      // Revert on error
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

  // US-023: Save persona assignment
  const handleSavePersona = async () => {
    if (!agentId) return
    setPersonaSaving(true)
    try {
      const payload: any = { use_custom: false }
      if (personaMode === 'predefined' && selectedPersonaId) {
        payload.persona_id = selectedPersonaId
      } else if (personaMode === 'custom' && customPersonaPrompt) {
        payload.custom_prompt = customPersonaPrompt
        payload.use_custom = true
      } else {
        // Clear persona
        payload.persona_id = null
        payload.custom_prompt = null
      }
      await apiClient.request(`/api/agents/${agentId}/persona`, {
        method: 'PUT',
        body: payload,
      })
      // Update current display state
      if (personaMode === 'predefined' && selectedPersonaId) {
        const persona = personas.find((p: any) => p.id === selectedPersonaId)
        setCurrentPersonaName(persona?.name || null)
        setCurrentPersonaPrompt(persona?.system_prompt || null)
      } else if (personaMode === 'custom') {
        setCurrentPersonaName(null)
        setCurrentPersonaPrompt(customPersonaPrompt)
      } else {
        setCurrentPersonaName(null)
        setCurrentPersonaPrompt(null)
      }
    } catch (err) {
      console.error('Failed to save persona:', err)
    } finally {
      setPersonaSaving(false)
    }
  }

  // PRD-55: Save heartbeat config
  const saveHeartbeatConfig = async () => {
    if (!agentId) return
    try {
      await apiClient.request(`/api/heartbeat/agents/${agentId}/config`, {
        method: 'PUT',
        body: heartbeatConfig as any,
      })
      toast.success('Heartbeat config saved')
    } catch (err) {
      console.error('Failed to save heartbeat config:', err)
      toast.error('Failed to save heartbeat config')
    }
  }

  // PRD-55: Run heartbeat now
  const runHeartbeatNow = async () => {
    if (!agentId) return
    setHeartbeatRunning(true)
    try {
      const result = await apiClient.request<any>(`/api/heartbeat/agents/${agentId}/run`, {
        method: 'POST',
      })
      setLastHeartbeatResult(result)
      toast.success('Heartbeat executed')
    } catch (err) {
      console.error('Failed to run heartbeat:', err)
      toast.error('Failed to run heartbeat')
    } finally {
      setHeartbeatRunning(false)
    }
  }

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
      const selectedCategory = formData.agent_type || 'custom'
      const mappedDbType = CATEGORY_TO_DB_MAP[selectedCategory] || 'custom'
      const originalMapsToCategory = DB_TO_CATEGORY_MAP[originalAgentType] || 'custom'
      const dbAgentType =
        (mappedDbType === 'custom' && originalMapsToCategory === selectedCategory)
          ? originalAgentType
          : mappedDbType

      const updatePayload = {
        name: formData.name,
        description: formData.description,
        job_title: (formData.job_title || '').trim(),
        agent_type: dbAgentType,
        marketplace_category: selectedCategory,
        tags,
        configuration: {
          category: selectedCategory,
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
        tool_ids: (formData.assigned_tools || []).filter((id: any) => id != null),
      }

      toast.loading('Saving configuration...', { id: 'agent-config-save' })

      // Save agent configuration
      await updateConfigMutation.mutateAsync({
        agentId: agentId.toString(),
        config: updatePayload
      })

      // PRD-15: Save model configuration
      if (formData.model_config) {
        await updateModelConfigMutation.mutateAsync({
          agentId: agentId,
          modelConfig: formData.model_config
        })
      }

      toast.success('Configuration saved!', { id: 'agent-config-save' })
      setHasChanges(false)

      if (onSave) {
        onSave(agentId, updatePayload)
      }

      onClose()

    } catch (err) {
      console.error('Error saving agent configuration:', err)
      toast.error(
        `Save failed: ${err instanceof Error ? err.message : String(err)}`,
        { id: 'agent-config-save' }
      )
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
        <Card className="glass-card card-glow w-full max-w-5xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3">
              {(() => {
                const category = agent ? DB_TO_CATEGORY_MAP[(agent as any)?.agent_type || 'custom'] || 'Custom' : 'Custom'
                const premiumIconName = iconMappings[(agent as any)?.marketplace_category] || iconMappings[(agent as any)?.configuration?.category] || iconMappings[(agent as any)?.agent_type] || iconMappings[category] || null
                return premiumIconName ? (
                  <PremiumIcon name={premiumIconName} size={28} className="text-primary" />
                ) : (
                  <Settings className="w-6 h-6 text-primary" />
                )
              })()}
              <div>
                <span className="text-xl">Agent <span className="gradient-text">Configuration</span></span>
                <p className="text-sm text-muted-foreground font-normal">
                  {(agent as any)?.name || 'Loading...'}
                </p>
              </div>
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSave}
                disabled={saving || !hasChanges}
              >
                <Save className="w-4 h-4 mr-2" />
                {saving ? 'Saving...' : 'Save Changes'}
              </Button>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
          </CardHeader>

          <CardContent className="overflow-y-auto p-6">
            {loading && (
              <LoadingState variant="spinner" label="Loading configuration…" className="py-12" />
            )}

            {error && (
              <ErrorState description={error} onRetry={() => refetchAgent()} className="py-12" />
            )}

            {agent && (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="w-full justify-start gap-1 bg-secondary/50">
                  <TabsTrigger value="general" className="flex items-center space-x-1">
                    <Info className="w-4 h-4" />
                    <span>General</span>
                  </TabsTrigger>
                  <TabsTrigger value="persona" className="flex items-center space-x-1">
                    <User className="w-4 h-4" />
                    <span>Persona</span>
                  </TabsTrigger>
                  <TabsTrigger value="resources" className="flex items-center space-x-1">
                    <Database className="w-4 h-4" />
                    <span>Resources</span>
                  </TabsTrigger>
                  <TabsTrigger value="plugins" className="flex items-center space-x-1">
                    <Sparkles className="w-4 h-4" />
                    <span>Capabilities</span>
                  </TabsTrigger>
                  <TabsTrigger value="model" className="flex items-center space-x-1">
                    <Bot className="w-4 h-4" />
                    <span>Model</span>
                  </TabsTrigger>
                  <TabsTrigger value="tools" className="flex items-center space-x-1">
                    <Wrench className="w-4 h-4" />
                    <span>Tools</span>
                  </TabsTrigger>
                  <TabsTrigger value="heartbeat" className="flex items-center space-x-1">
                    <Activity className="w-3 h-3" />
                    <span>Heartbeat</span>
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
                        <Label htmlFor="job_title">Job Title</Label>
                        <Input
                          id="job_title"
                          value={formData.job_title || ''}
                          onChange={(e) => updateFormData('job_title', e.target.value)}
                          placeholder="e.g. Lead Intelligence, Code Watchdog, Memory Keeper"
                          maxLength={120}
                        />
                        <p className="text-xs text-muted-foreground">
                          Short role label shown under the agent&apos;s name on the roster card
                          (e.g. &quot;Research &middot; Lead Intelligence&quot;).
                        </p>
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
                          value={formData.agent_type || 'custom'}
                          onValueChange={(value) => updateFormData('agent_type', value)}
                        >
                          <SelectTrigger>
                            {formData.agent_type ? (
                              <div className="flex items-center gap-2">
                                {(() => {
                                  const selected = AGENT_CATEGORIES.find(c => c.id === formData.agent_type)
                                  if (!selected) return <SelectValue />
                                  const premiumName = iconMappings[selected.id]
                                  const FallbackIcon = selected.icon
                                  return (
                                    <>
                                      {premiumName ? (
                                        <PremiumIcon name={premiumName} size={16} className={selected.color} />
                                      ) : (
                                        <FallbackIcon className={`w-4 h-4 ${selected.color}`} />
                                      )}
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
                              const premiumName = iconMappings[cat.id]
                              const FallbackIcon = cat.icon
                              return (
                                <SelectItem key={cat.id} value={cat.id}>
                                  <div className="flex items-center gap-2">
                                    {premiumName ? (
                                      <PremiumIcon name={premiumName} size={16} className={cat.color} />
                                    ) : (
                                      <FallbackIcon className={`w-4 h-4 ${cat.color}`} />
                                    )}
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

                {/* US-023: Persona Tab */}
                <TabsContent value="persona" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <User className="h-5 w-5 text-[hsl(var(--agent))]" />
                        Agent Persona
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Give your agent a personality and voice
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Persona Mode Selection */}
                      <div role="radiogroup" aria-label="Persona mode" className="grid grid-cols-3 gap-3">
                        <button
                          type="button"
                          role="radio"
                          aria-checked={personaMode === 'none'}
                          className={`p-4 rounded-lg border cursor-pointer transition-all text-center ${personaMode === 'none'
                            ? 'border-primary bg-primary/10'
                            : 'border-border/50 hover:border-primary/30'
                            }`}
                          onClick={() => {
                            setPersonaMode('none')
                            setSelectedPersonaId(null)
                            setCustomPersonaPrompt('')
                          }}
                        >
                          <Bot className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                          <div className="font-medium text-sm">No Persona</div>
                          <div className="text-xs text-muted-foreground mt-1">Default behavior</div>
                        </button>

                        <button
                          type="button"
                          role="radio"
                          aria-checked={personaMode === 'predefined'}
                          className={`p-4 rounded-lg border cursor-pointer transition-all text-center ${personaMode === 'predefined'
                            ? 'border-primary bg-primary/10'
                            : 'border-border/50 hover:border-primary/30'
                            }`}
                          onClick={() => setPersonaMode('predefined')}
                        >
                          <User className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                          <div className="font-medium text-sm">Predefined</div>
                          <div className="text-xs text-muted-foreground mt-1">Choose a persona</div>
                        </button>

                        <button
                          type="button"
                          role="radio"
                          aria-checked={personaMode === 'custom'}
                          className={`p-4 rounded-lg border cursor-pointer transition-all text-center ${personaMode === 'custom'
                            ? 'border-primary bg-primary/10'
                            : 'border-border/50 hover:border-primary/30'
                            }`}
                          onClick={() => setPersonaMode('custom')}
                        >
                          <PenLine className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                          <div className="font-medium text-sm">Custom</div>
                          <div className="text-xs text-muted-foreground mt-1">Write your own</div>
                        </button>
                      </div>

                      {/* Predefined Persona Selection */}
                      {personaMode === 'predefined' && (
                        <div className="space-y-4">
                          {/* Category Filter */}
                          <div>
                            <Label>Filter by Category</Label>
                            <Select
                              value={personaCategoryFilter}
                              onValueChange={setPersonaCategoryFilter}
                            >
                              <SelectTrigger className="bg-secondary/50">
                                <SelectValue placeholder="All categories" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="all">All Categories</SelectItem>
                                {[...new Set(personas.map((p: any) => p.category).filter(Boolean))].map((cat: string) => (
                                  <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Persona List */}
                          {personasLoading ? (
                            <div className="space-y-2">
                              {[1, 2, 3].map(i => (
                                <div key={i} className="h-16 bg-secondary/20 animate-pulse rounded-lg" />
                              ))}
                            </div>
                          ) : (
                            <div className="space-y-2 max-h-[30vh] overflow-y-auto pr-1">
                              {personas
                                .filter((p: any) => personaCategoryFilter === 'all' || p.category === personaCategoryFilter)
                                .map((persona: any) => (
                                  <div
                                    key={persona.id}
                                    className={`p-3 rounded-lg border cursor-pointer transition-all ${selectedPersonaId === persona.id
                                      ? 'border-primary bg-primary/10'
                                      : 'border-border/50 hover:border-primary/30'
                                      }`}
                                    onClick={() => setSelectedPersonaId(persona.id)}
                                  >
                                    <div className="flex items-center justify-between">
                                      <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2">
                                          <span className="font-medium">{persona.name}</span>
                                          {persona.category && (
                                            <Badge variant="outline" className="text-xs">{persona.category}</Badge>
                                          )}
                                        </div>
                                        {persona.voice_description && (
                                          <p className="text-xs text-muted-foreground mt-1 truncate">
                                            {persona.voice_description}
                                          </p>
                                        )}
                                        <p className="text-xs text-muted-foreground mt-0.5">
                                          Temperature: {persona.suggested_temperature}
                                        </p>
                                      </div>
                                      <div className="flex items-center gap-2 ml-2">
                                        {selectedPersonaId === persona.id && (
                                          <Badge className="bg-primary text-primary-foreground">Selected</Badge>
                                        )}
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={(e) => {
                                            e.stopPropagation()
                                            setExpandedPersonaId(
                                              expandedPersonaId === persona.id ? null : persona.id
                                            )
                                          }}
                                        >
                                          {expandedPersonaId === persona.id ? (
                                            <ChevronUp className="w-4 h-4" />
                                          ) : (
                                            <ChevronDown className="w-4 h-4" />
                                          )}
                                        </Button>
                                      </div>
                                    </div>

                                    {/* Expandable System Prompt Preview */}
                                    {expandedPersonaId === persona.id && persona.system_prompt && (
                                      <div className="mt-3 pt-3 border-t border-border/30">
                                        <Label className="text-xs">System Prompt</Label>
                                        <pre className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap bg-secondary/30 rounded p-2 max-h-[150px] overflow-y-auto">
                                          {persona.system_prompt}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                ))}
                              {personas.filter((p: any) => personaCategoryFilter === 'all' || p.category === personaCategoryFilter).length === 0 && (
                                <div className="text-center py-6 text-muted-foreground">
                                  No personas found{personaCategoryFilter !== 'all' ? ` in category "${personaCategoryFilter}"` : ''}.
                                </div>
                              )}
                            </div>
                          )}

                          {/* Tip */}
                          {selectedPersonaId && (
                            <p className="text-xs text-muted-foreground italic">
                              Tip: Select a predefined persona and switch to &quot;Custom&quot; to pre-fill for editing.
                            </p>
                          )}
                        </div>
                      )}

                      {/* Custom Persona */}
                      {personaMode === 'custom' && (
                        <div className="space-y-3">
                          <div>
                            <Label htmlFor="custom-persona-config">Custom Persona Prompt</Label>
                            <Textarea
                              id="custom-persona-config"
                              placeholder="Describe the agent's personality, communication style, expertise, and behavioral guidelines..."
                              value={customPersonaPrompt}
                              onChange={(e) => setCustomPersonaPrompt(e.target.value)}
                              className="bg-secondary/50 min-h-[200px] font-mono text-sm"
                            />
                            <p className="text-xs text-muted-foreground mt-1">
                              This prompt will be prepended to the agent&apos;s system message.
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Current Status & Save */}
                      <div className="flex justify-between items-center pt-4 border-t border-border/30">
                        <div>
                          <p className="font-medium text-sm">
                            {personaMode === 'none' && 'No persona selected'}
                            {personaMode === 'predefined' && (selectedPersonaId
                              ? `Persona: ${personas.find((p: any) => p.id === selectedPersonaId)?.name || 'Selected'}`
                              : 'Select a persona above')}
                            {personaMode === 'custom' && (customPersonaPrompt
                              ? `Custom persona (${customPersonaPrompt.length} chars)`
                              : 'Write a custom persona above')}
                          </p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleSavePersona}
                          disabled={personaSaving}
                          className="hover:border-[hsl(var(--agent))]/50"
                        >
                          {personaSaving ? (
                            <>
                              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                              Saving...
                            </>
                          ) : (
                            <>
                              <Save className="w-4 h-4 mr-2" />
                              Save Persona
                            </>
                          )}
                        </Button>
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

                {/* PRD-42: Plugins Tab */}
                <TabsContent value="plugins" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-primary" />
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
                        <div className="space-y-3">
                          {workspacePlugins.map((plugin: any) => {
                            const isAssigned = assignedPluginIds.has(plugin.plugin_id)
                            return (
                              <div
                                key={plugin.plugin_id}
                                className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors ${isAssigned
                                  ? 'bg-primary/5 border-primary/30'
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
                                          <Badge variant="secondary" className="text-xs text-[hsl(var(--success))] border-[hsl(var(--success))]/30">
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
                          <Sparkles className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Capabilities Available</h3>
                          <p className="text-muted-foreground text-sm mb-4">
                            No capabilities are enabled for this workspace yet.
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

                  {/* PRD-71: Skill Assignment */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Zap className="h-5 w-5 text-primary" />
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
                        <div className="space-y-3">
                          {workspaceSkills.map((skill: any) => {
                            const isAssigned = assignedSkillIds.has(skill.skill_id)
                            return (
                              <div
                                key={skill.skill_id}
                                className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors ${isAssigned
                                  ? 'bg-primary/5 border-primary/30'
                                  : 'bg-background/50 border-border/50'
                                  }`}
                              >
                                <Checkbox
                                  id={`skill-${skill.skill_id}`}
                                  checked={isAssigned}
                                  onCheckedChange={() => toggleSkillAssignment(skill.skill_id)}
                                  disabled={skillsSaving}
                                  className="mt-1"
                                />
                                <div className="flex-1 min-w-0">
                                  <Label htmlFor={`skill-${skill.skill_id}`} className="cursor-pointer">
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
                </TabsContent>

                <TabsContent value="tools" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Wrench className="h-5 w-5 text-[hsl(var(--info))]" />
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
                        <Bot className="h-5 w-5 text-[hsl(var(--agent))]" />
                        Model Configuration
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Select and configure the LLM model for this agent
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Model Selection */}
                      <ModelSelector
                        value={formData.model_config?.model_id || LLM_DEFAULTS.model_id}
                        onChange={(modelId) => updateModelConfig('model_id', modelId)}
                        agentType={formData.agent_type}
                      />

                      <Separator />

                      {/* Temperature */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="temperature" className="flex items-center gap-1">Temperature <InlineHelp id="agents.config.model.temperature" size="sm" /></Label>
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
                          <Label htmlFor="max-tokens" className="flex items-center gap-1">Max Output Tokens <InlineHelp id="agents.config.model.max_tokens" size="sm" /></Label>
                          <span className="text-sm text-muted-foreground">
                            {formData.model_config?.max_tokens || 2000}
                          </span>
                        </div>
                        <Slider
                          id="max-tokens"
                          value={[formData.model_config?.max_tokens || 2000]}
                          onValueChange={([value]) => updateModelConfig('max_tokens', value)}
                          min={100}
                          max={16384}
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
                            <Label htmlFor="top-p" className="text-xs flex items-center gap-1">Top P (Nucleus Sampling) <InlineHelp id="agents.config.model.top_p" size="sm" /></Label>
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
                            <Label htmlFor="frequency-penalty" className="text-xs flex items-center gap-1">Frequency Penalty <InlineHelp id="agents.config.model.frequency_penalty" size="sm" /></Label>
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
                            <Label htmlFor="presence-penalty" className="text-xs flex items-center gap-1">Presence Penalty <InlineHelp id="agents.config.model.presence_penalty" size="sm" /></Label>
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
                        <Label htmlFor="fallback-model" className="flex items-center gap-1">Fallback Model (Optional) <InlineHelp id="agents.config.model.fallback_model" size="sm" /></Label>
                        <Select
                          value={formData.model_config?.fallback_model_id || 'none'}
                          onValueChange={(value) => updateModelConfig('fallback_model_id', value === 'none' ? null : value)}
                        >
                          <SelectTrigger id="fallback-model" className="bg-background/50 border-border">
                            <SelectValue placeholder="Select fallback model..." />
                          </SelectTrigger>
                          <SelectContent className="bg-popover border-border">
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

                {/* PRD-55: Heartbeat Configuration Tab */}
                <TabsContent value="heartbeat" className="space-y-6 mt-6 max-h-[60vh] overflow-y-auto pr-2">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Activity className="h-5 w-5 text-[hsl(var(--info))]" />
                        Agent Heartbeat
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Configure periodic autonomous check-ins for this agent
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Enable Heartbeat */}
                      <div className="flex items-center justify-between">
                        <div>
                          <Label className="flex items-center gap-1">Enable Heartbeat <InlineHelp id="agents.config.heartbeat.enable" size="sm" /></Label>
                          <p className="text-xs text-muted-foreground">Agent will periodically wake up and check its environment</p>
                        </div>
                        <Switch
                          checked={heartbeatConfig.enabled}
                          onCheckedChange={(v) => setHeartbeatConfig(prev => ({ ...prev, enabled: v }))}
                        />
                      </div>

                      {heartbeatConfig.enabled && (
                        <>
                          {/* Interval */}
                          <div className="space-y-2">
                            <Label className="flex items-center gap-1">Interval <InlineHelp id="agents.config.heartbeat.interval" size="sm" /></Label>
                            <Select
                              value={String(heartbeatConfig.interval_minutes)}
                              onValueChange={(v) => setHeartbeatConfig(prev => ({ ...prev, interval_minutes: Number(v) }))}
                            >
                              <SelectTrigger><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="15">Every 15 minutes</SelectItem>
                                <SelectItem value="30">Every 30 minutes</SelectItem>
                                <SelectItem value="60">Every hour</SelectItem>
                                <SelectItem value="120">Every 2 hours</SelectItem>
                                <SelectItem value="240">Every 4 hours</SelectItem>
                                <SelectItem value="480">Every 8 hours</SelectItem>
                                <SelectItem value="1440">Daily</SelectItem>
                                <SelectItem value="10080">Weekly</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Active Hours */}
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <Label className="flex items-center gap-1">Active Hours <InlineHelp id="agents.config.heartbeat.active_hours" size="sm" /></Label>
                              <div className="flex items-center gap-2">
                                <Checkbox
                                  id="inherit-hours"
                                  checked={heartbeatConfig.inherit_active_hours}
                                  onCheckedChange={(v) => setHeartbeatConfig(prev => ({ ...prev, inherit_active_hours: !!v }))}
                                />
                                <Label htmlFor="inherit-hours" className="text-xs cursor-pointer">Inherit from orchestrator</Label>
                              </div>
                            </div>
                            {!heartbeatConfig.inherit_active_hours && (
                              <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-1">
                                  <Label className="text-xs">From</Label>
                                  <Input
                                    type="time"
                                    value={heartbeatConfig.active_hours_start}
                                    onChange={(e) => setHeartbeatConfig(prev => ({ ...prev, active_hours_start: e.target.value }))}
                                  />
                                </div>
                                <div className="space-y-1">
                                  <Label className="text-xs">Until</Label>
                                  <Input
                                    type="time"
                                    value={heartbeatConfig.active_hours_end}
                                    onChange={(e) => setHeartbeatConfig(prev => ({ ...prev, active_hours_end: e.target.value }))}
                                  />
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Heartbeat Prompt */}
                          <div className="space-y-2">
                            <Label className="flex items-center gap-1">Heartbeat Prompt <InlineHelp id="agents.config.heartbeat.prompt" size="sm" /></Label>
                            <Textarea
                              value={heartbeatConfig.prompt}
                              onChange={(e) => setHeartbeatConfig(prev => ({ ...prev, prompt: e.target.value }))}
                              placeholder="What should this agent check during heartbeat? e.g. Check for new emails, review pending tasks..."
                              className="bg-secondary/50 min-h-[100px]"
                            />
                          </div>

                          {/* Auto-Act */}
                          <div className="flex items-center justify-between">
                            <div>
                              <Label className="flex items-center gap-1">Auto-Act on Findings <InlineHelp id="agents.config.heartbeat.auto_act" size="sm" /></Label>
                              <p className="text-xs text-muted-foreground">Agent can take action based on heartbeat results</p>
                            </div>
                            <Switch
                              checked={heartbeatConfig.auto_act}
                              onCheckedChange={(v) => setHeartbeatConfig(prev => ({ ...prev, auto_act: v }))}
                            />
                          </div>

                          {/* Report To */}
                          <div className="space-y-2">
                            <Label className="flex items-center gap-1">Report To <InlineHelp id="agents.config.heartbeat.report_to" size="sm" /></Label>
                            <Select
                              value={heartbeatConfig.report_to}
                              onValueChange={(v) => setHeartbeatConfig(prev => ({ ...prev, report_to: v }))}
                            >
                              <SelectTrigger><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="orchestrator">DB only</SelectItem>
                                <SelectItem value="auto">Auto (assign task)</SelectItem>
                                {connectedIntegrations.map((i) => (
                                  <SelectItem key={i.key} value={i.platform}>
                                    {i.platform.charAt(0).toUpperCase() + i.platform.slice(1)}
                                  </SelectItem>
                                ))}
                                <SelectItem value="webhook">Webhook URL</SelectItem>
                              </SelectContent>
                            </Select>
                            <p className="text-xs text-muted-foreground">
                              {heartbeatConfig.report_to === 'orchestrator' && 'Results stored in DB. Auto only sees them if asked.'}
                              {heartbeatConfig.report_to === 'auto' && 'Creates a board task assigned to Auto. Auto picks it up on its next tick.'}
                              {heartbeatConfig.report_to === 'telegram' && 'Results sent to your Telegram chat'}
                              {heartbeatConfig.report_to === 'slack' && 'Results sent to your Slack channel'}
                              {heartbeatConfig.report_to === 'webhook' && 'Results POSTed as JSON to your URL'}
                            </p>
                          </div>

                          {/* Webhook URL (shown when webhook selected) */}
                          {heartbeatConfig.report_to === 'webhook' && (
                            <div className="space-y-2">
                              <Label>Webhook URL</Label>
                              <Input
                                type="url"
                                value={heartbeatConfig.webhook_url || ''}
                                onChange={(e) => setHeartbeatConfig(prev => ({ ...prev, webhook_url: e.target.value }))}
                                placeholder="https://hooks.slack.com/... or any endpoint"
                                className="bg-secondary/50"
                              />
                            </div>
                          )}

                          {/* Slack channel ID (only shown for Slack — Telegram auto-resolves) */}
                          {heartbeatConfig.report_to === 'slack' && (
                            <div className="space-y-2">
                              <Label>Slack Channel ID</Label>
                              <Input
                                value={heartbeatConfig.channel_id || ''}
                                onChange={(e) => setHeartbeatConfig(prev => ({ ...prev, channel_id: e.target.value }))}
                                placeholder="e.g. C01ABCDEF"
                                className="bg-secondary/50"
                              />
                            </div>
                          )}
                        </>
                      )}

                      {/* Actions */}
                      <div className="flex gap-2 pt-4 border-t border-border/30">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={saveHeartbeatConfig}
                          className="flex-1"
                        >
                          <Save className="w-4 h-4 mr-2" />
                          Save Heartbeat Config
                        </Button>
                        {heartbeatConfig.enabled && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={runHeartbeatNow}
                            disabled={heartbeatRunning}
                          >
                            {heartbeatRunning ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                            Run Now
                          </Button>
                        )}
                      </div>

                      {/* Last Heartbeat Result */}
                      {lastHeartbeatResult && (
                        <div className="space-y-2 pt-4 border-t border-border/30">
                          <Label className="text-xs">Last Heartbeat Result</Label>
                          <div className="p-3 rounded-lg bg-secondary/30 text-sm">
                            <div className="flex items-center gap-2 mb-2">
                              <span className={`h-2 w-2 rounded-full ${lastHeartbeatResult.status === 'success' ? 'bg-success' : 'bg-destructive'}`} />
                              <span className="text-xs text-muted-foreground">
                                {lastHeartbeatResult.created_at ? new Date(lastHeartbeatResult.created_at).toLocaleString() : 'Unknown time'}
                              </span>
                            </div>
                            <p className="text-xs whitespace-pre-wrap">{lastHeartbeatResult.summary || lastHeartbeatResult.result || 'No details available'}</p>
                          </div>
                        </div>
                      )}
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
