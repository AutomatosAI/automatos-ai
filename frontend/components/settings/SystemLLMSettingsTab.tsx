/**
 * Orchestrator Settings Tab Component (PRD-55)
 * =============================================
 *
 * Expanded from "System LLM Settings" to full Orchestrator Soul Designer.
 * 4 sections:
 * 1. LLM Configuration (existing, enhanced with thinking level)
 * 2. Soul & Personality (NEW)
 * 3. Heartbeat (NEW)
 * 4. What Automatos Knows (NEW, read-only)
 */

import React, { useState, useMemo, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import {
  Save, RotateCcw, Brain, Zap, Settings, Heart, Sparkles,
  ChevronDown, Clock, MessageSquare, AlertCircle, Loader2,
  Database, Eye
} from 'lucide-react'
import { toast } from 'sonner'
import {
  SystemSetting,
  getSettingsForCategory,
  bulkUpdateSettings,
  resetSettingsToDefaults,
} from '@/lib/api/system-settings'
import { useWorkspaceModels } from '@/hooks/use-model-api'
import { apiClient } from '@/lib/api-client'

interface SystemLLMSettingsTabProps {
  settings?: SystemSetting[]
  onSave?: (updates: Record<string, string>) => void
  saving?: boolean
  onReset?: () => void
}

interface MemoryStats {
  total_memories: number
  by_type: Record<string, number>
  by_level: Record<string, number>
  agents_with_memories: number
  recent_24h: number
  recent: Array<{
    id: string
    type: string
    level: string
    preview: string
    created_at: string | null
  }>
}

interface OrchestratorConfig {
  personality_mode: string
  custom_soul: string
  communication_style: string
  proactive_level: string
  thinking_level: string
  heartbeat: {
    enabled: boolean
    interval_minutes: number
    active_hours_start: string
    active_hours_end: string
    timezone: string
    checklist: string
    notification_channel: string
  }
}

const PERSONALITY_PRESETS: Record<string, { label: string; description: string }> = {
  friendly: { label: 'Friendly', description: 'Warm, approachable, celebrates wins with you' },
  professional: { label: 'Professional', description: 'Polished, structured, enterprise-appropriate' },
  technical: { label: 'Technical', description: 'Precise, detailed, developer-focused' },
  custom: { label: 'Custom', description: 'Write your own personality prompt' },
}

const COMMUNICATION_STYLES: Record<string, { label: string; description: string }> = {
  concise: { label: 'Concise', description: 'Short, direct answers' },
  balanced: { label: 'Balanced', description: 'Clear answers with helpful context' },
  detailed: { label: 'Detailed', description: 'Thorough explanations with examples' },
}

const PROACTIVE_LEVELS: Record<string, { label: string; description: string }> = {
  silent: { label: 'Silent', description: 'Heartbeat runs but never notifies' },
  notify: { label: 'Notify Only', description: 'Reports findings but takes no action' },
  act_notify: { label: 'Act & Notify', description: 'Takes action and tells you what it did' },
  autonomous: { label: 'Fully Autonomous', description: 'Acts independently, reports summary' },
}

const THINKING_LEVELS: Record<string, string> = {
  off: 'Off',
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
}

export default function SystemLLMSettingsTab({
  settings: externalSettings,
  onSave: externalOnSave,
  saving: externalSaving,
  onReset: externalOnReset
}: SystemLLMSettingsTabProps) {
  // Self-loading state (used when no props passed — top-level mode)
  const isStandalone = !externalSettings
  const [selfSettings, setSelfSettings] = useState<SystemSetting[]>([])
  const [selfLoading, setSelfLoading] = useState(isStandalone)
  const [selfSaving, setSelfSaving] = useState(false)

  const settings = externalSettings ?? selfSettings
  const saving = externalSaving ?? selfSaving

  // Existing LLM form data (from system_settings table)
  const [formData, setFormData] = useState<Record<string, string>>({})

  // Orchestrator config (from workspace.settings.orchestrator)
  const [orchConfig, setOrchConfig] = useState<OrchestratorConfig | null>(null)
  const [orchLoading, setOrchLoading] = useState(true)
  const [orchSaving, setOrchSaving] = useState(false)

  // Memory stats (from workspace API)
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null)
  const [memoryLoading, setMemoryLoading] = useState(true)

  // Collapsible section states
  const [llmOpen, setLlmOpen] = useState(true)
  const [soulOpen, setSoulOpen] = useState(true)
  const [heartbeatOpen, setHeartbeatOpen] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(false)

  // Load workspace-installed models (same catalog agents use)
  const { data: allModels = [], isLoading: modelsLoading } = useWorkspaceModels()

  const selectedProvider = formData.llm_provider || ''

  const availableModels = useMemo(() => {
    if (!Array.isArray(allModels)) return []
    if (!selectedProvider) return allModels
    // For aggregator providers (openrouter), also include aggregator-tier models
    // whose underlying provider differs (e.g. Claude Opus 4.6 via OpenRouter has
    // provider="anthropic" but tier="aggregator")
    const isAggregator = selectedProvider === 'openrouter'
    return allModels.filter((model: any) =>
      model.provider === selectedProvider ||
      (isAggregator && model.tier === 'aggregator')
    )
  }, [allModels, selectedProvider])

  // Self-load settings when in standalone mode
  useEffect(() => {
    if (!isStandalone) return
    const loadSelfSettings = async () => {
      try {
        setSelfLoading(true)
        const data = await getSettingsForCategory('orchestrator_llm')
        setSelfSettings(data)
      } catch (err) {
        console.error('Failed to self-load orchestrator_llm settings:', err)
        toast.error('Failed to load LLM settings')
      } finally {
        setSelfLoading(false)
      }
    }
    loadSelfSettings()
  }, [isStandalone])

  // Initialize LLM form data from system settings
  useEffect(() => {
    const initialData: Record<string, string> = {}
    settings.forEach(setting => {
      initialData[setting.key] = setting.value !== null && setting.value !== undefined
        ? setting.value
        : (setting.default_value || '')
    })
    setFormData(initialData)
  }, [settings])

  // Load orchestrator config from workspace API
  useEffect(() => {
    const loadOrchConfig = async () => {
      try {
        setOrchLoading(true)
        const data = await apiClient.request<OrchestratorConfig>('/api/workspaces/current/orchestrator')
        setOrchConfig(data)
      } catch (err) {
        console.error('Failed to load orchestrator config:', err)
        // Use defaults if endpoint not available
        setOrchConfig({
          personality_mode: 'friendly',
          custom_soul: '',
          communication_style: 'balanced',
          proactive_level: 'notify',
          thinking_level: 'medium',
          heartbeat: {
            enabled: false,
            interval_minutes: 30,
            active_hours_start: '08:00',
            active_hours_end: '20:00',
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
            checklist: '- Check agent health status\n- Review pending webhook failures\n- Summarize today\'s activity',
            notification_channel: 'in_app',
          },
        })
      } finally {
        setOrchLoading(false)
      }
    }
    loadOrchConfig()
  }, [])

  // Load memory stats
  useEffect(() => {
    const loadMemoryStats = async () => {
      try {
        setMemoryLoading(true)
        const data = await apiClient.request<MemoryStats>('/api/workspaces/current/memory-stats')
        setMemoryStats(data)
      } catch (err) {
        console.error('Failed to load memory stats:', err)
        setMemoryStats(null)
      } finally {
        setMemoryLoading(false)
      }
    }
    loadMemoryStats()
  }, [])

  const handleInputChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleOrchChange = (key: string, value: string | boolean) => {
    if (!orchConfig) return
    setOrchConfig({ ...orchConfig, [key]: value })
  }

  const handleHeartbeatChange = (key: string, value: string | boolean | number) => {
    if (!orchConfig) return
    setOrchConfig({
      ...orchConfig,
      heartbeat: { ...orchConfig.heartbeat, [key]: value },
    })
  }

  const handleSaveLLM = async () => {
    if (externalOnSave) {
      externalOnSave(formData)
      return
    }
    // Self-save mode
    try {
      setSelfSaving(true)
      const bulkUpdates = settings
        .map(setting => ({
          id: setting.id,
          value: formData[setting.key] !== undefined ? formData[setting.key] : (setting.value || setting.default_value || '')
        }))
      if (bulkUpdates.length > 0) {
        await bulkUpdateSettings(bulkUpdates)
        toast.success('LLM settings saved')
        // Reload
        const data = await getSettingsForCategory('orchestrator_llm')
        setSelfSettings(data)
      }
    } catch (err) {
      console.error('Failed to save LLM settings:', err)
      toast.error('Failed to save LLM settings')
    } finally {
      setSelfSaving(false)
    }
  }

  const handleSaveOrchestrator = async () => {
    if (!orchConfig) return
    try {
      setOrchSaving(true)
      await apiClient.request('/api/workspaces/current/orchestrator', {
        method: 'PUT',
        body: JSON.stringify(orchConfig),
      })
      toast.success('Orchestrator settings saved')
    } catch (err) {
      console.error('Failed to save orchestrator settings:', err)
      toast.error('Failed to save orchestrator settings')
    } finally {
      setOrchSaving(false)
    }
  }

  const handleReset = async () => {
    const defaultData: Record<string, string> = {}
    settings.forEach(setting => {
      defaultData[setting.key] = setting.default_value || ''
    })
    setFormData(defaultData)
    if (externalOnReset) {
      externalOnReset()
      return
    }
    // Self-reset mode
    try {
      setSelfSaving(true)
      await resetSettingsToDefaults('orchestrator_llm')
      toast.success('LLM settings reset to defaults')
      const data = await getSettingsForCategory('orchestrator_llm')
      setSelfSettings(data)
    } catch (err) {
      console.error('Failed to reset LLM settings:', err)
      toast.error('Failed to reset LLM settings')
    } finally {
      setSelfSaving(false)
    }
  }

  const getSetting = (key: string) => settings.find(s => s.key === key)

  const soulTokenEstimate = orchConfig?.custom_soul
    ? Math.ceil(orchConfig.custom_soul.length / 4)
    : 0

  if (selfLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading orchestrator settings...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Section 1: LLM Configuration */}
      <Collapsible open={llmOpen} onOpenChange={setLlmOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  <div>
                    <CardTitle>LLM Configuration</CardTitle>
                    <CardDescription>
                      Provider, model, and performance settings for the orchestrator
                    </CardDescription>
                  </div>
                </div>
                <ChevronDown className={`h-5 w-5 transition-transform ${llmOpen ? 'rotate-180' : ''}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              {/* Provider & Model */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="llm_provider">LLM Provider</Label>
                  <Select
                    value={formData.llm_provider || ''}
                    onValueChange={(value) => handleInputChange('llm_provider', value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select LLM provider" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai">OpenAI</SelectItem>
                      <SelectItem value="anthropic">Anthropic</SelectItem>
                      <SelectItem value="google">Google</SelectItem>
                      <SelectItem value="openrouter">OpenRouter</SelectItem>
                      <SelectItem value="deepseek">DeepSeek</SelectItem>
                      <SelectItem value="azure">Azure OpenAI</SelectItem>
                      <SelectItem value="bedrock">AWS Bedrock</SelectItem>
                      <SelectItem value="grok">Grok / xAI</SelectItem>
                      <SelectItem value="cohere">Cohere</SelectItem>
                      <SelectItem value="huggingface">HuggingFace (Free/Testing)</SelectItem>
                      <SelectItem value="local">Local Model</SelectItem>
                    </SelectContent>
                  </Select>
                  {getSetting('llm_provider')?.is_required && (
                    <Badge variant="destructive" className="text-xs">Required</Badge>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="llm_model">LLM Model</Label>
                  <Select
                    value={formData.llm_model || ''}
                    onValueChange={(value) => {
                      handleInputChange('llm_model', value)
                      // If user picks an aggregator model, auto-switch provider to openrouter
                      // (aggregator models use OpenRouter paths like "anthropic/claude-opus-4.6")
                      const picked = allModels.find((m: any) => m.model_id === value)
                      if (picked && (picked as any).tier === 'aggregator' && formData.llm_provider !== 'openrouter') {
                        handleInputChange('llm_provider', 'openrouter')
                      }
                    }}
                    disabled={modelsLoading || !selectedProvider}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={
                        modelsLoading
                          ? "Loading models..."
                          : !selectedProvider
                            ? "Select provider first"
                            : "Select model"
                      } />
                    </SelectTrigger>
                    <SelectContent>
                      {availableModels.length > 0 ? (
                        availableModels.map((model: { model_id: string; display_name: string; context_window: number }) => (
                          <SelectItem key={model.model_id} value={model.model_id}>
                            {model.display_name}
                            {model.context_window >= 100000 && ` (${(model.context_window / 1000).toFixed(0)}K context)`}
                          </SelectItem>
                        ))
                      ) : (
                        !modelsLoading && selectedProvider && (
                          <SelectItem value="" disabled>No models available for {selectedProvider}</SelectItem>
                        )
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Thinking Level + Temperature */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Thinking Level</Label>
                  <Select
                    value={orchConfig?.thinking_level || 'medium'}
                    onValueChange={(v) => handleOrchChange('thinking_level', v)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(THINKING_LEVELS).map(([value, label]) => (
                        <SelectItem key={value} value={value}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Extended reasoning budget</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="llm_temperature">Temperature</Label>
                  <Input
                    id="llm_temperature"
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={formData.llm_temperature || ''}
                    onChange={(e) => handleInputChange('llm_temperature', e.target.value)}
                    placeholder="0.7"
                  />
                  <p className="text-xs text-muted-foreground">0 = deterministic, 2 = creative</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="llm_max_tokens">Max Tokens</Label>
                  <Input
                    id="llm_max_tokens"
                    type="number"
                    min="1"
                    max="32000"
                    value={formData.llm_max_tokens || ''}
                    onChange={(e) => handleInputChange('llm_max_tokens', e.target.value)}
                    placeholder="2000"
                  />
                  <p className="text-xs text-muted-foreground">Max tokens in response</p>
                </div>
              </div>

              {/* Performance Settings (collapsed) */}
              <Collapsible>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2">
                    <Zap className="h-4 w-4" />
                    Advanced Performance Settings
                    <ChevronDown className="h-3 w-3" />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Request Timeout (seconds)</Label>
                      <Input
                        type="number" min="5" max="300"
                        value={formData.llm_timeout || '30'}
                        onChange={(e) => handleInputChange('llm_timeout', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Retry Attempts</Label>
                      <Input
                        type="number" min="0" max="5"
                        value={formData.llm_retry_attempts || '3'}
                        onChange={(e) => handleInputChange('llm_retry_attempts', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Concurrent Requests</Label>
                      <Input
                        type="number" min="1" max="10"
                        value={formData.llm_concurrent_requests || '5'}
                        onChange={(e) => handleInputChange('llm_concurrent_requests', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Cache TTL (seconds)</Label>
                      <Input
                        type="number" min="0" max="3600"
                        value={formData.llm_cache_ttl || '300'}
                        onChange={(e) => handleInputChange('llm_cache_ttl', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Top P</Label>
                      <Input
                        type="number" step="0.1" min="0" max="1"
                        value={formData.llm_top_p || '1'}
                        onChange={(e) => handleInputChange('llm_top_p', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Frequency Penalty</Label>
                      <Input
                        type="number" step="0.1" min="-2" max="2"
                        value={formData.llm_frequency_penalty || '0'}
                        onChange={(e) => handleInputChange('llm_frequency_penalty', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Presence Penalty</Label>
                      <Input
                        type="number" step="0.1" min="-2" max="2"
                        value={formData.llm_presence_penalty || '0'}
                        onChange={(e) => handleInputChange('llm_presence_penalty', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Stop Sequences</Label>
                      <Input
                        value={formData.llm_stop_sequences || ''}
                        onChange={(e) => handleInputChange('llm_stop_sequences', e.target.value)}
                        placeholder="\n\n, ###, END"
                      />
                    </div>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {/* LLM Save */}
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={handleReset} disabled={saving}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button size="sm" onClick={handleSaveLLM} disabled={saving}>
                  <Save className="h-4 w-4 mr-2" />
                  Save LLM Settings
                </Button>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Section 2: Soul & Personality */}
      <Collapsible open={soulOpen} onOpenChange={setSoulOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  <div>
                    <CardTitle>Soul & Personality</CardTitle>
                    <CardDescription>
                      Define how your AI assistant communicates and behaves
                    </CardDescription>
                  </div>
                </div>
                <ChevronDown className={`h-5 w-5 transition-transform ${soulOpen ? 'rotate-180' : ''}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              {orchLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading personality settings...
                </div>
              ) : orchConfig ? (
                <>
                  {/* Personality Mode */}
                  <div className="space-y-3">
                    <Label>Personality Mode</Label>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(PERSONALITY_PRESETS).map(([key, preset]) => (
                        <button
                          key={key}
                          onClick={() => handleOrchChange('personality_mode', key)}
                          className={`p-3 rounded-lg border text-left transition-colors ${
                            orchConfig.personality_mode === key
                              ? 'border-primary bg-primary/10'
                              : 'border-border hover:border-primary/50'
                          }`}
                        >
                          <div className="font-medium text-sm">{preset.label}</div>
                          <div className="text-xs text-muted-foreground mt-1">{preset.description}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Custom Soul Editor (shown when custom mode selected) */}
                  {orchConfig.personality_mode === 'custom' && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Custom Soul Prompt</Label>
                        <span className="text-xs text-muted-foreground">
                          ~{soulTokenEstimate} tokens
                        </span>
                      </div>
                      <Textarea
                        value={orchConfig.custom_soul}
                        onChange={(e) => handleOrchChange('custom_soul', e.target.value)}
                        placeholder="Define your AI assistant's personality, tone, and behavior..."
                        rows={8}
                        className="font-mono text-sm"
                      />
                      <p className="text-xs text-muted-foreground">
                        Markdown supported. This overrides the default personality and is injected into every orchestrator call.
                      </p>
                    </div>
                  )}

                  {/* Communication Style */}
                  <div className="space-y-3">
                    <Label>Communication Style</Label>
                    <div className="grid grid-cols-3 gap-3">
                      {Object.entries(COMMUNICATION_STYLES).map(([key, style]) => (
                        <button
                          key={key}
                          onClick={() => handleOrchChange('communication_style', key)}
                          className={`p-3 rounded-lg border text-left transition-colors ${
                            orchConfig.communication_style === key
                              ? 'border-primary bg-primary/10'
                              : 'border-border hover:border-primary/50'
                          }`}
                        >
                          <div className="font-medium text-sm">{style.label}</div>
                          <div className="text-xs text-muted-foreground mt-1">{style.description}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Proactive Level */}
                  <div className="space-y-3">
                    <Label>Proactive Behavior Level</Label>
                    <Select
                      value={orchConfig.proactive_level}
                      onValueChange={(v) => handleOrchChange('proactive_level', v)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(PROACTIVE_LEVELS).map(([key, level]) => (
                          <SelectItem key={key} value={key}>
                            {level.label} — {level.description}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Controls what the orchestrator does when its heartbeat finds something
                    </p>
                  </div>
                </>
              ) : null}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Section 3: Heartbeat */}
      <Collapsible open={heartbeatOpen} onOpenChange={setHeartbeatOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Heart className="h-5 w-5" />
                  <div>
                    <CardTitle>Orchestrator Heartbeat</CardTitle>
                    <CardDescription>
                      Periodic check-ins — your AI team standup runs automatically
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {orchConfig?.heartbeat.enabled && (
                    <Badge variant="default" className="text-xs">Active</Badge>
                  )}
                  <ChevronDown className={`h-5 w-5 transition-transform ${heartbeatOpen ? 'rotate-180' : ''}`} />
                </div>
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              {orchConfig ? (
                <>
                  {/* Enable Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Enable Heartbeat</Label>
                      <p className="text-xs text-muted-foreground">
                        The orchestrator wakes up periodically to check on your agents and tasks
                      </p>
                    </div>
                    <Switch
                      checked={orchConfig.heartbeat.enabled}
                      onCheckedChange={(checked) => handleHeartbeatChange('enabled', checked)}
                    />
                  </div>

                  {orchConfig.heartbeat.enabled && (
                    <>
                      {/* Interval & Hours */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="space-y-2">
                          <Label className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            Interval
                          </Label>
                          <Select
                            value={String(orchConfig.heartbeat.interval_minutes)}
                            onValueChange={(v) => handleHeartbeatChange('interval_minutes', parseInt(v))}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="15">Every 15 minutes</SelectItem>
                              <SelectItem value="30">Every 30 minutes</SelectItem>
                              <SelectItem value="60">Every hour</SelectItem>
                              <SelectItem value="120">Every 2 hours</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label>Active From</Label>
                          <Input
                            type="time"
                            value={orchConfig.heartbeat.active_hours_start}
                            onChange={(e) => handleHeartbeatChange('active_hours_start', e.target.value)}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Active Until</Label>
                          <Input
                            type="time"
                            value={orchConfig.heartbeat.active_hours_end}
                            onChange={(e) => handleHeartbeatChange('active_hours_end', e.target.value)}
                          />
                        </div>
                      </div>

                      {/* Timezone & Notification */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Timezone</Label>
                          <Input
                            value={orchConfig.heartbeat.timezone}
                            onChange={(e) => handleHeartbeatChange('timezone', e.target.value)}
                            placeholder="America/New_York"
                          />
                          <p className="text-xs text-muted-foreground">IANA timezone (auto-detected from browser)</p>
                        </div>

                        <div className="space-y-2">
                          <Label className="flex items-center gap-1">
                            <MessageSquare className="h-3 w-3" />
                            Notification Channel
                          </Label>
                          <Select
                            value={orchConfig.heartbeat.notification_channel}
                            onValueChange={(v) => handleHeartbeatChange('notification_channel', v)}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="in_app">In-App Notification</SelectItem>
                              <SelectItem value="email">Email</SelectItem>
                              <SelectItem value="webhook">Webhook URL</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      {/* Checklist */}
                      <div className="space-y-2">
                        <Label>Heartbeat Checklist</Label>
                        <Textarea
                          value={orchConfig.heartbeat.checklist}
                          onChange={(e) => handleHeartbeatChange('checklist', e.target.value)}
                          placeholder="- Check agent health status&#10;- Review pending tasks&#10;- Summarize today's activity"
                          rows={5}
                          className="font-mono text-sm"
                        />
                        <p className="text-xs text-muted-foreground">
                          What should the orchestrator check each heartbeat? One item per line.
                        </p>
                      </div>
                    </>
                  )}
                </>
              ) : null}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Section 4: What Automatos Knows */}
      <Collapsible open={memoryOpen} onOpenChange={setMemoryOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  <div>
                    <CardTitle>What Automatos Knows</CardTitle>
                    <CardDescription>
                      Memory stats and knowledge stored across your workspace
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {memoryStats && memoryStats.total_memories > 0 && (
                    <Badge variant="secondary" className="text-xs">
                      {memoryStats.total_memories} memories
                    </Badge>
                  )}
                  <ChevronDown className={`h-5 w-5 transition-transform ${memoryOpen ? 'rotate-180' : ''}`} />
                </div>
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              {memoryLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading memory stats...
                </div>
              ) : memoryStats ? (
                <>
                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="rounded-lg border p-3 text-center">
                      <div className="text-2xl font-bold">{memoryStats.total_memories}</div>
                      <div className="text-xs text-muted-foreground">Total Memories</div>
                    </div>
                    <div className="rounded-lg border p-3 text-center">
                      <div className="text-2xl font-bold">{memoryStats.agents_with_memories}</div>
                      <div className="text-xs text-muted-foreground">Agents with Memory</div>
                    </div>
                    <div className="rounded-lg border p-3 text-center">
                      <div className="text-2xl font-bold">{memoryStats.recent_24h}</div>
                      <div className="text-xs text-muted-foreground">Last 24 Hours</div>
                    </div>
                    <div className="rounded-lg border p-3 text-center">
                      <div className="text-2xl font-bold">{Object.keys(memoryStats.by_type).length}</div>
                      <div className="text-xs text-muted-foreground">Memory Types</div>
                    </div>
                  </div>

                  {/* By Type Breakdown */}
                  {Object.keys(memoryStats.by_type).length > 0 && (
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">
                        <Eye className="h-3 w-3" />
                        By Type
                      </Label>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(memoryStats.by_type).map(([type, count]) => (
                          <Badge key={type} variant="outline">
                            {type}: {count}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* By Level Breakdown */}
                  {Object.keys(memoryStats.by_level).length > 0 && (
                    <div className="space-y-2">
                      <Label>By Level</Label>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(memoryStats.by_level).map(([level, count]) => (
                          <Badge key={level} variant="outline">
                            {level}: {count}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recent Memories */}
                  {memoryStats.recent.length > 0 && (
                    <div className="space-y-2">
                      <Label>Recent Memories</Label>
                      <div className="space-y-2">
                        {memoryStats.recent.map((mem) => (
                          <div key={mem.id} className="rounded border p-2 text-sm">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="secondary" className="text-xs">{mem.type}</Badge>
                              <Badge variant="outline" className="text-xs">{mem.level}</Badge>
                              {mem.created_at && (
                                <span className="text-xs text-muted-foreground ml-auto">
                                  {new Date(mem.created_at).toLocaleDateString()}
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-muted-foreground truncate">{mem.preview}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {memoryStats.total_memories === 0 && (
                    <div className="text-center py-6 text-muted-foreground">
                      <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No memories stored yet. Start chatting to build knowledge!</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  Memory stats unavailable
                </div>
              )}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Save Orchestrator Settings (Soul + Heartbeat) */}
      <div className="flex justify-end">
        <Button onClick={handleSaveOrchestrator} disabled={orchSaving || orchLoading}>
          {orchSaving ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Save className="h-4 w-4 mr-2" />
          )}
          Save Orchestrator Settings
        </Button>
      </div>
    </div>
  )
}
