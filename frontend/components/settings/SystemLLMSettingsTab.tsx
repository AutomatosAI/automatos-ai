/**
 * Orchestrator Settings Tab Component (PRD-55)
 * =============================================
 *
 * Expanded from "System LLM Settings" to full Orchestrator Soul Designer.
 * 4 sections:
 * 1. LLM Configuration (existing, enhanced with thinking level)
 * 2. Soul & Personality (NEW)
 * 3. Heartbeat (NEW)
 * 4. HARNESS — Self-Optimizing Organization Loop (PRD-121)
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
  Play, Shield, Calendar, Volume2
} from 'lucide-react'
import { toast } from 'sonner'
import { InlineHelp } from '@/components/ui/help-tooltip'
import {
  SystemSetting,
  getSettingsForCategory,
} from '@/lib/api/system-settings'
import { useWorkspaceModels } from '@/hooks/use-model-api'
import { apiClient } from '@/lib/api-client'

interface SystemLLMSettingsTabProps {
  settings?: SystemSetting[]
  onSave?: (updates: Record<string, string>) => void
  saving?: boolean
  onReset?: () => void
}

interface LLMConfig {
  provider: string
  model_id: string
  temperature: number
  max_tokens: number
  top_p: number
  frequency_penalty: number
  presence_penalty: number
  stop: string[] | null
  timeout: number | null
  fallback_model_id: string | null
}

interface OrchestratorConfig {
  personality_mode: string
  custom_soul: string
  communication_style: string
  proactive_level: string
  thinking_level: string
  llm?: LLMConfig
  heartbeat: {
    enabled: boolean
    interval_minutes: number
    active_hours_start: string
    active_hours_end: string
    timezone: string
    checklist: string
    notification_channel: string
    channel_id?: string
  }
  harness: {
    enabled: boolean
    schedule: string
    mode: string
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

  // Legacy LLM form data (from system_settings table — performance settings only)
  const [formData, setFormData] = useState<Record<string, string>>({})

  // Orchestrator config (from workspace.settings.orchestrator + Auto agent)
  const [orchConfig, setOrchConfig] = useState<OrchestratorConfig | null>(null)
  const [orchLoading, setOrchLoading] = useState(true)
  const [orchSaving, setOrchSaving] = useState(false)


  // Collapsible section states
  const [llmOpen, setLlmOpen] = useState(true)
  const [soulOpen, setSoulOpen] = useState(true)
  const [heartbeatOpen, setHeartbeatOpen] = useState(false)
  const [harnessOpen, setHarnessOpen] = useState(false)

  // Voice profiles for Auto's voice
  const [voiceProfiles, setVoiceProfiles] = useState<Array<{ id: string; name: string; provider: string; voice_id: string }>>([])
  const [selectedVoiceProfileId, setSelectedVoiceProfileId] = useState<string | null>(null)

  // Heartbeat: connected channels, run now, last result
  const [connectedChannels, setConnectedChannels] = useState<Array<{ key: string; platform: string }>>([])
  const [heartbeatRunning, setHeartbeatRunning] = useState(false)
  const [lastHeartbeatResult, setLastHeartbeatResult] = useState<any>(null)

  // Load workspace-installed models (same catalog agents use)
  const { data: allModels = [], isLoading: modelsLoading } = useWorkspaceModels()

  const selectedProvider = orchConfig?.llm?.provider || ''

  const availableModels = useMemo(() => {
    if (!Array.isArray(allModels)) return []
    if (!selectedProvider) return allModels
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
          llm: {
            provider: 'openrouter',
            model_id: 'google/gemini-2.5-flash',
            temperature: 0.7,
            max_tokens: 4000,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop: null,
            timeout: null,
            fallback_model_id: null,
          },
          harness: {
            enabled: false,
            schedule: 'weekly',
            mode: 'full_auto',
          },
        })
      } finally {
        setOrchLoading(false)
      }
    }
    loadOrchConfig()
  }, [])


  // Load voice profiles for Auto's voice selector
  useEffect(() => {
    apiClient.request<{ items: any[]; total: number }>('/api/voice/profiles')
      .then((data) => {
        if (data?.items && Array.isArray(data.items)) setVoiceProfiles(data.items)
      })
      .catch(() => {})
  }, [])

  // Sync selected voice profile when orchestrator config loads
  useEffect(() => {
    if (orchConfig && (orchConfig as any).voice_profile_id) {
      setSelectedVoiceProfileId((orchConfig as any).voice_profile_id)
    }
  }, [orchConfig])

  // Load connected channels for notification dropdown + last heartbeat result
  useEffect(() => {
    // Connected channels
    Promise.all([
      apiClient.request<any>('/api/workspaces/current/integrations').catch(() => ({})),
      apiClient.request<any>('/api/channels').catch(() => []),
    ]).then(([integrations, channels]) => {
      const found: Array<{ key: string; platform: string }> = []
      const seen = new Set<string>()
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
      if (Array.isArray(channels)) {
        for (const ch of channels) {
          if (!seen.has(ch.platform)) {
            found.push({ key: `channel:${ch.id}`, platform: ch.platform })
            seen.add(ch.platform)
          }
        }
      }
      setConnectedChannels(found)
    })

    // Last heartbeat result
    apiClient.request<any>('/api/heartbeat/orchestrator/history?limit=1')
      .then((data) => {
        if (data?.results?.[0]) setLastHeartbeatResult(data.results[0])
      })
      .catch(() => {})
  }, [])

  const handleInputChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleOrchChange = (key: string, value: string | boolean) => {
    if (!orchConfig) return
    setOrchConfig({ ...orchConfig, [key]: value })
  }

  const handleLLMChange = (key: string, value: string | number | null) => {
    if (!orchConfig) return
    const current = orchConfig.llm || { provider: '', model_id: '', temperature: 0.7, max_tokens: 4000, top_p: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0, stop: null, timeout: null, fallback_model_id: null }
    setOrchConfig({ ...orchConfig, llm: { ...current, [key]: value } })
  }

  const handleHeartbeatChange = (key: string, value: string | boolean | number) => {
    if (!orchConfig) return
    setOrchConfig({
      ...orchConfig,
      heartbeat: { ...orchConfig.heartbeat, [key]: value },
    })
  }

  const handleHarnessChange = (key: string, value: string | boolean) => {
    if (!orchConfig) return
    setOrchConfig({
      ...orchConfig,
      harness: { ...orchConfig.harness, [key]: value },
    })
  }

  const runHeartbeatNow = async () => {
    setHeartbeatRunning(true)
    try {
      const result = await apiClient.request<any>('/api/heartbeat/orchestrator/run', {
        method: 'POST',
      })
      setLastHeartbeatResult(result)
      toast.success('Orchestrator heartbeat executed')
    } catch (err) {
      console.error('Failed to run orchestrator heartbeat:', err)
      toast.error('Failed to run orchestrator heartbeat')
    } finally {
      setHeartbeatRunning(false)
    }
  }

  const handleSaveLLM = async () => {
    if (!orchConfig?.llm) return
    try {
      setSelfSaving(true)
      // Save LLM through the orchestrator endpoint → Auto agent model_config
      await apiClient.request('/api/workspaces/current/orchestrator', {
        method: 'PUT',
        body: JSON.stringify({ llm: orchConfig.llm }),
      })
      toast.success('LLM settings saved')
    } catch (err) {
      toast.error('Failed to save LLM settings')
    } finally {
      setSelfSaving(false)
    }
  }

  const handleSaveOrchestrator = async () => {
    if (!orchConfig) return
    try {
      setOrchSaving(true)
      // Parse channel key to extract platform + channel_id for heartbeat delivery
      const hb = { ...orchConfig.heartbeat }
      const channelKey = hb.notification_channel
      if (channelKey?.startsWith('channel:')) {
        const channelId = channelKey.replace('channel:', '')
        const found = connectedChannels.find(ch => ch.key === channelKey)
        hb.notification_channel = found?.platform || channelKey
        hb.channel_id = channelId
      } else if (channelKey === 'in_app' || channelKey === 'webhook') {
        // Built-in channels, no channel_id needed
      } else {
        // Direct platform name (e.g. "telegram" from workspace integrations)
        hb.notification_channel = channelKey
      }
      const payload = { ...orchConfig, heartbeat: hb, voice_profile_id: selectedVoiceProfileId || null }
      await apiClient.request('/api/workspaces/current/orchestrator', {
        method: 'PUT',
        body: JSON.stringify(payload),
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
    // Reset LLM to defaults by saving empty llm object — backend will use config defaults
    try {
      setSelfSaving(true)
      await apiClient.request('/api/workspaces/current/orchestrator', {
        method: 'PUT',
        body: JSON.stringify({ llm: { provider: 'openrouter', model_id: 'google/gemini-2.5-flash', temperature: 0.7, max_tokens: 4000, top_p: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0, stop: null, timeout: null, fallback_model_id: null } }),
      })
      // Reload orchestrator config to get fresh values
      const data = await apiClient.request<OrchestratorConfig>('/api/workspaces/current/orchestrator')
      setOrchConfig(data)
      toast.success('LLM settings reset to defaults')
    } catch (err) {
      toast.error('Failed to reset LLM settings')
    } finally {
      setSelfSaving(false)
    }
  }

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
                  <Label htmlFor="llm_provider" className="flex items-center gap-1">LLM Provider <InlineHelp id="settings.llm.provider" size="sm" /></Label>
                  <Select
                    value={orchConfig?.llm?.provider || ''}
                    onValueChange={(value) => handleLLMChange('provider', value)}
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
                  {!orchConfig?.llm?.provider && (
                    <Badge variant="destructive" className="text-xs">Required</Badge>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="llm_model" className="flex items-center gap-1">LLM Model <InlineHelp id="settings.llm.model" size="sm" /></Label>
                  <Select
                    value={orchConfig?.llm?.model_id || ''}
                    onValueChange={(value) => {
                      handleLLMChange('model_id', value)
                      // If user picks an aggregator model, auto-switch provider to openrouter
                      const picked = allModels.find((m: any) => m.model_id === value)
                      if (picked && (picked as any).tier === 'aggregator' && orchConfig?.llm?.provider !== 'openrouter') {
                        handleLLMChange('provider', 'openrouter')
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
                          <SelectItem value="__no_models__" disabled>No models available for {selectedProvider}</SelectItem>
                        )
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Thinking Level + Temperature */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label className="flex items-center gap-1">Thinking Level <InlineHelp id="settings.llm.thinking_level" size="sm" /></Label>
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
                  <Label htmlFor="llm_temperature" className="flex items-center gap-1">Temperature <InlineHelp id="settings.llm.temperature" size="sm" /></Label>
                  <Input
                    id="llm_temperature"
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={orchConfig?.llm?.temperature ?? ''}
                    onChange={(e) => handleLLMChange('temperature', parseFloat(e.target.value) || 0)}
                    placeholder="0.7"
                  />
                  <p className="text-xs text-muted-foreground">0 = deterministic, 2 = creative</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="llm_max_tokens" className="flex items-center gap-1">Max Tokens <InlineHelp id="settings.llm.max_tokens" size="sm" /></Label>
                  <Input
                    id="llm_max_tokens"
                    type="number"
                    min="1"
                    max="32000"
                    value={orchConfig?.llm?.max_tokens ?? ''}
                    onChange={(e) => handleLLMChange('max_tokens', parseInt(e.target.value) || 0)}
                    placeholder="4000"
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
                      <Label className="flex items-center gap-1">Request Timeout (seconds) <InlineHelp id="settings.llm.timeout" size="sm" /></Label>
                      <Input
                        type="number" min="5" max="300"
                        value={orchConfig?.llm?.timeout ?? 180}
                        onChange={(e) => handleLLMChange('timeout', parseInt(e.target.value) || null)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Retry Attempts <InlineHelp id="settings.llm.retry_attempts" size="sm" /></Label>
                      <Input
                        type="number" min="0" max="5"
                        value={formData.llm_retry_attempts || '3'}
                        onChange={(e) => handleInputChange('llm_retry_attempts', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Concurrent Requests <InlineHelp id="settings.llm.concurrent_requests" size="sm" /></Label>
                      <Input
                        type="number" min="1" max="10"
                        value={formData.llm_concurrent_requests || '5'}
                        onChange={(e) => handleInputChange('llm_concurrent_requests', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Cache TTL (seconds) <InlineHelp id="settings.llm.cache_ttl" size="sm" /></Label>
                      <Input
                        type="number" min="0" max="3600"
                        value={formData.llm_cache_ttl || '300'}
                        onChange={(e) => handleInputChange('llm_cache_ttl', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Top P <InlineHelp id="settings.llm.top_p" size="sm" /></Label>
                      <Input
                        type="number" step="0.1" min="0" max="1"
                        value={orchConfig?.llm?.top_p ?? 1}
                        onChange={(e) => handleLLMChange('top_p', parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Frequency Penalty <InlineHelp id="settings.llm.frequency_penalty" size="sm" /></Label>
                      <Input
                        type="number" step="0.1" min="-2" max="2"
                        value={orchConfig?.llm?.frequency_penalty ?? 0}
                        onChange={(e) => handleLLMChange('frequency_penalty', parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Presence Penalty <InlineHelp id="settings.llm.presence_penalty" size="sm" /></Label>
                      <Input
                        type="number" step="0.1" min="-2" max="2"
                        value={orchConfig?.llm?.presence_penalty ?? 0}
                        onChange={(e) => handleLLMChange('presence_penalty', parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-1">Stop Sequences <InlineHelp id="settings.llm.stop_sequences" size="sm" /></Label>
                      <Input
                        value={(orchConfig?.llm?.stop || []).join(', ')}
                        onChange={(e) => {
                          const val = e.target.value
                          const sequences = val ? val.split(',').map((s: string) => s.trim()).filter(Boolean) : null
                          handleLLMChange('stop', sequences)
                        }}
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
                    <Label className="flex items-center gap-1">Personality Mode <InlineHelp id="settings.soul.personality_mode" size="sm" /></Label>
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
                        <Label className="flex items-center gap-1">Custom Soul Prompt <InlineHelp id="settings.soul.custom_soul" size="sm" /></Label>
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
                    <Label className="flex items-center gap-1">Communication Style <InlineHelp id="settings.soul.communication_style" size="sm" /></Label>
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
                    <Label className="flex items-center gap-1">Proactive Behavior Level <InlineHelp id="settings.soul.proactive_level" size="sm" /></Label>
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

                  {/* Voice Profile */}
                  <div className="space-y-3">
                    <Label className="flex items-center gap-1">
                      <Volume2 className="h-4 w-4" />
                      Auto&apos;s Voice
                    </Label>
                    <Select
                      value={selectedVoiceProfileId || 'none'}
                      onValueChange={(v) => setSelectedVoiceProfileId(v === 'none' ? null : v)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="No voice assigned" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">No voice (text only)</SelectItem>
                        {voiceProfiles.map((vp) => (
                          <SelectItem key={vp.id} value={vp.id}>
                            {vp.name} ({vp.provider})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Voice used for Auto&apos;s TTS responses. Create voice profiles in Settings &gt; Voices.
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
                      <Label className="flex items-center gap-1">Enable Heartbeat <InlineHelp id="settings.heartbeat.enable" size="sm" /></Label>
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
                            <InlineHelp id="settings.heartbeat.interval" size="sm" />
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
                              <SelectItem value="240">Every 4 hours</SelectItem>
                              <SelectItem value="480">Every 8 hours</SelectItem>
                              <SelectItem value="1440">Daily</SelectItem>
                              <SelectItem value="10080">Weekly</SelectItem>
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
                              {connectedChannels.map((ch) => (
                                <SelectItem key={ch.key} value={ch.key}>
                                  {ch.platform.charAt(0).toUpperCase() + ch.platform.slice(1)}
                                </SelectItem>
                              ))}
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

                      {/* Run Now & Last Result */}
                      <div className="flex gap-2 pt-4 border-t border-border/30">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={runHeartbeatNow}
                          disabled={heartbeatRunning}
                        >
                          {heartbeatRunning ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                          Run Now
                        </Button>
                      </div>

                      {lastHeartbeatResult && (
                        <div className="space-y-2 pt-4 border-t border-border/30">
                          <Label className="text-xs">Last Heartbeat Result</Label>
                          <div className="p-3 rounded-lg bg-secondary/30 text-sm">
                            <div className="flex items-center gap-2 mb-2">
                              <span className={`h-2 w-2 rounded-full ${lastHeartbeatResult.status === 'success' ? 'bg-green-500' : 'bg-red-500'}`} />
                              <span className="font-medium text-xs">{lastHeartbeatResult.status}</span>
                              <span className="text-xs text-muted-foreground">
                                {lastHeartbeatResult.created_at ? new Date(lastHeartbeatResult.created_at).toLocaleString() : ''}
                              </span>
                              {lastHeartbeatResult.tokens_used > 0 && (
                                <Badge variant="outline" className="text-xs ml-auto">{lastHeartbeatResult.tokens_used} tokens</Badge>
                              )}
                            </div>
                            <p className="text-xs whitespace-pre-wrap">
                              {lastHeartbeatResult.findings?.find((f: any) => f.check === 'llm_analysis')?.detail
                                || lastHeartbeatResult.findings?.[0]?.detail
                                || 'No details available'}
                            </p>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </>
              ) : null}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Section 4: HARNESS — Self-Optimizing Organization Loop */}
      <Collapsible open={harnessOpen} onOpenChange={setHarnessOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  <div>
                    <CardTitle>HARNESS</CardTitle>
                    <CardDescription>
                      Self-optimizing loop that tunes agent configurations based on performance data
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {orchConfig?.harness?.enabled && (
                    <Badge variant="default" className="text-xs">Active</Badge>
                  )}
                  <ChevronDown className={`h-5 w-5 transition-transform ${harnessOpen ? 'rotate-180' : ''}`} />
                </div>
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              {orchConfig ? (
                <>
                  {/* Description */}
                  <div className="rounded-lg bg-secondary/30 p-4 text-sm text-muted-foreground">
                    <p className="mb-2">
                      <strong className="text-foreground">HARNESS</strong> (Holistic Agent Review, Normalization, Evaluation &amp; Self-Shaping)
                      periodically collects org-wide metrics, diagnoses regressions, and prescribes configuration
                      changes to keep your agent fleet running optimally.
                    </p>
                    <p>
                      Safe changes (low risk) can be applied automatically, or you can require manual approval
                      for every change. Risky changes are always queued as board tasks for your review.
                    </p>
                  </div>

                  {/* Enable Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Enable HARNESS</Label>
                      <p className="text-xs text-muted-foreground">
                        When disabled, no automatic optimization runs will occur
                      </p>
                    </div>
                    <Switch
                      checked={orchConfig.harness?.enabled ?? false}
                      onCheckedChange={(checked) => handleHarnessChange('enabled', checked)}
                    />
                  </div>

                  {orchConfig.harness?.enabled && (
                    <>
                      {/* Schedule & Mode */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            Run Schedule
                          </Label>
                          <Select
                            value={orchConfig.harness.schedule || 'weekly'}
                            onValueChange={(v) => handleHarnessChange('schedule', v)}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="weekly">Weekly (Sunday 2 AM UTC)</SelectItem>
                              <SelectItem value="biweekly">Biweekly</SelectItem>
                              <SelectItem value="monthly">Monthly</SelectItem>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-muted-foreground">
                            How often HARNESS analyzes your organization and proposes changes
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label className="flex items-center gap-1">
                            <Shield className="h-3 w-3" />
                            Approval Mode
                          </Label>
                          <Select
                            value={orchConfig.harness.mode || 'full_auto'}
                            onValueChange={(v) => handleHarnessChange('mode', v)}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="full_auto">Full Auto — low-risk changes applied automatically</SelectItem>
                              <SelectItem value="manual">Manual — all changes require your approval</SelectItem>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-muted-foreground">
                            {orchConfig.harness.mode === 'manual'
                              ? 'Every proposed change will be queued as a board task for your review'
                              : 'Changes with risk score 1-2 are applied automatically; risk 3+ queued for review'}
                          </p>
                        </div>
                      </div>
                    </>
                  )}
                </>
              ) : null}
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
