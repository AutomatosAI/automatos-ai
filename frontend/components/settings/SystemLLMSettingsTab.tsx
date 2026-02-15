/**
 * System LLM Settings Tab Component
 * ==================================
 * 
 * Manages the base LLM configuration for the entire system
 * (orchestrator, workflows, agents, etc.)
 */

import React, { useState, useEffect, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Save, RotateCcw, Brain, Zap, Settings, Activity, Sparkles, Loader2 } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import { useModels } from '@/hooks/use-model-api'

interface SystemLLMSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function SystemLLMSettingsTab({
  settings,
  onSave,
  saving,
  onReset
}: SystemLLMSettingsTabProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})

  // PRD-55: Soul & Personality state
  const [soulSettings, setSoulSettings] = useState({
    personality_mode: 'friendly',
    custom_soul: '',
    communication_style: 'balanced',
    proactive_level: 'notify',
    thinking_level: 'medium'
  })
  const [savingSoul, setSavingSoul] = useState(false)

  // PRD-55: Heartbeat state
  const [heartbeatSettings, setHeartbeatSettings] = useState({
    enabled: false,
    interval_minutes: 30,
    active_hours_start: '08:00',
    active_hours_end: '20:00',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
    checklist: '- Check pending tasks\n- Review unrouted events\n- Summarize today\'s activity'
  })
  const [savingHeartbeat, setSavingHeartbeat] = useState(false)

  // PRD-55: Memory state
  const [memoryStats, setMemoryStats] = useState<any>({ total: 0, agents: 0, recent24h: 0, recent: [] })
  const [memoryLoading, setMemoryLoading] = useState(false)

  // Load models from API
  const { data: allModels = [], isLoading: modelsLoading } = useModels(undefined, 'active')

  // Get selected provider to filter models
  const selectedProvider = formData.llm_provider || ''

  // Filter models by selected provider
  const availableModels = useMemo(() => {
    if (!Array.isArray(allModels)) return []
    if (!selectedProvider) return allModels
    return allModels.filter((model: { provider: string }) => model.provider === selectedProvider)
  }, [allModels, selectedProvider])

  // Initialize form data from settings
  React.useEffect(() => {
    const initialData: Record<string, string> = {}
    settings.forEach(setting => {
      // Use saved value if it exists, otherwise use default, but don't treat empty string as falsy
      initialData[setting.key] = setting.value !== null && setting.value !== undefined
        ? setting.value
        : (setting.default_value || '')
    })
    setFormData(initialData)
  }, [settings])

  // PRD-55: Load soul, heartbeat, and memory settings
  useEffect(() => {
    // Load soul + heartbeat settings from workspace
    fetch('/api/workspaces/current/settings')
      .then(r => r.json())
      .then(data => {
        const orch = data?.settings?.orchestrator || {};
        if (orch.personality_mode) setSoulSettings(s => ({ ...s, ...orch }));
        if (orch.heartbeat) setHeartbeatSettings(s => ({ ...s, ...orch.heartbeat }));
      })
      .catch(() => {});

    // Load memory stats
    setMemoryLoading(true);
    fetch('/api/workspaces/current/memory-stats')
      .then(r => r.json())
      .then(data => setMemoryStats(data))
      .catch(() => {})
      .finally(() => setMemoryLoading(false));
  }, []);

  // PRD-55: Save soul settings
  const saveSoulSettings = async () => {
    setSavingSoul(true);
    try {
      await fetch('/api/workspaces/current/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ orchestrator: soulSettings })
      });
    } finally {
      setSavingSoul(false);
    }
  };

  // PRD-55: Save heartbeat settings
  const saveHeartbeatSettings = async () => {
    setSavingHeartbeat(true);
    try {
      await fetch('/api/workspaces/current/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ orchestrator: { heartbeat: heartbeatSettings } })
      });
    } finally {
      setSavingHeartbeat(false);
    }
  };

  const handleInputChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    onSave(formData)
  }

  const handleReset = () => {
    const defaultData: Record<string, string> = {}
    settings.forEach(setting => {
      defaultData[setting.key] = setting.default_value || ''
    })
    setFormData(defaultData)
    onReset()
  }

  const getSetting = (key: string) => settings.find(s => s.key === key)

  return (
    <div className="space-y-6">
      {/* System LLM Provider Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            System LLM Provider Configuration
          </CardTitle>
          <CardDescription>
            Configure the default LLM provider and model for workflows, orchestration, and agent operations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
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
                onValueChange={(value) => handleInputChange('llm_model', value)}
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
              {getSetting('llm_model')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
              {!selectedProvider && (
                <p className="text-xs text-muted-foreground">
                  Please select a provider first to see available models
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* LLM Parameters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            LLM Parameters
          </CardTitle>
          <CardDescription>
            Fine-tune LLM behavior and performance settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              <p className="text-xs text-muted-foreground">
                Controls randomness (0 = deterministic, 2 = very random)
              </p>
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
              <p className="text-xs text-muted-foreground">
                Maximum tokens in response (1-32000)
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Performance Settings
          </CardTitle>
          <CardDescription>
            Configure LLM performance and optimization settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="llm_timeout">Request Timeout (seconds)</Label>
              <Input
                id="llm_timeout"
                type="number"
                min="5"
                max="300"
                value={formData.llm_timeout || '30'}
                onChange={(e) => handleInputChange('llm_timeout', e.target.value)}
                placeholder="30"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm_retry_attempts">Retry Attempts</Label>
              <Input
                id="llm_retry_attempts"
                type="number"
                min="0"
                max="5"
                value={formData.llm_retry_attempts || '3'}
                onChange={(e) => handleInputChange('llm_retry_attempts', e.target.value)}
                placeholder="3"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm_concurrent_requests">Concurrent Requests</Label>
              <Input
                id="llm_concurrent_requests"
                type="number"
                min="1"
                max="10"
                value={formData.llm_concurrent_requests || '5'}
                onChange={(e) => handleInputChange('llm_concurrent_requests', e.target.value)}
                placeholder="5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm_cache_ttl">Cache TTL (seconds)</Label>
              <Input
                id="llm_cache_ttl"
                type="number"
                min="0"
                max="3600"
                value={formData.llm_cache_ttl || '300'}
                onChange={(e) => handleInputChange('llm_cache_ttl', e.target.value)}
                placeholder="300"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model-Specific Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Model-Specific Settings
          </CardTitle>
          <CardDescription>
            Advanced settings specific to the selected model
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="llm_top_p">Top P</Label>
              <Input
                id="llm_top_p"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={formData.llm_top_p || '1'}
                onChange={(e) => handleInputChange('llm_top_p', e.target.value)}
                placeholder="1"
              />
              <p className="text-xs text-muted-foreground">
                Nucleus sampling parameter (0-1)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm_frequency_penalty">Frequency Penalty</Label>
              <Input
                id="llm_frequency_penalty"
                type="number"
                step="0.1"
                min="-2"
                max="2"
                value={formData.llm_frequency_penalty || '0'}
                onChange={(e) => handleInputChange('llm_frequency_penalty', e.target.value)}
                placeholder="0"
              />
              <p className="text-xs text-muted-foreground">
                Reduces repetition (-2 to 2)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm_presence_penalty">Presence Penalty</Label>
              <Input
                id="llm_presence_penalty"
                type="number"
                step="0.1"
                min="-2"
                max="2"
                value={formData.llm_presence_penalty || '0'}
                onChange={(e) => handleInputChange('llm_presence_penalty', e.target.value)}
                placeholder="0"
              />
              <p className="text-xs text-muted-foreground">
                Encourages new topics (-2 to 2)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm_stop_sequences">Stop Sequences</Label>
              <Input
                id="llm_stop_sequences"
                value={formData.llm_stop_sequences || ''}
                onChange={(e) => handleInputChange('llm_stop_sequences', e.target.value)}
                placeholder="\n\n, ###, END"
              />
              <p className="text-xs text-muted-foreground">
                Comma-separated stop sequences
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* PRD-55: Soul & Personality */}
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Sparkles className="h-5 w-5" /> Soul & Personality</CardTitle>
          <CardDescription>Configure your orchestrator&apos;s personality and communication style</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Personality Mode */}
          <div className="space-y-2">
            <Label>Personality Mode</Label>
            <Select value={soulSettings.personality_mode} onValueChange={(v) => setSoulSettings(s => ({...s, personality_mode: v}))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="friendly">Friendly</SelectItem>
                <SelectItem value="professional">Professional</SelectItem>
                <SelectItem value="technical">Technical</SelectItem>
                <SelectItem value="custom">Custom Soul</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Custom Soul Editor */}
          {soulSettings.personality_mode === 'custom' && (
            <div className="space-y-2">
              <Label>Custom Soul</Label>
              <textarea className="w-full min-h-[120px] rounded-md border bg-background px-3 py-2 text-sm"
                value={soulSettings.custom_soul}
                onChange={(e) => setSoulSettings(s => ({...s, custom_soul: e.target.value}))}
                placeholder="Describe your AI's personality, tone, and behavior..."
              />
              <p className="text-xs text-muted-foreground">{soulSettings.custom_soul?.length || 0} chars (~{Math.ceil((soulSettings.custom_soul?.length || 0) / 4)} tokens)</p>
            </div>
          )}

          {/* Communication Style */}
          <div className="space-y-2">
            <Label>Communication Style</Label>
            <Select value={soulSettings.communication_style} onValueChange={(v) => setSoulSettings(s => ({...s, communication_style: v}))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="concise">Concise</SelectItem>
                <SelectItem value="balanced">Balanced</SelectItem>
                <SelectItem value="detailed">Detailed</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Proactive Level */}
          <div className="space-y-2">
            <Label>Proactive Level</Label>
            <Select value={soulSettings.proactive_level} onValueChange={(v) => setSoulSettings(s => ({...s, proactive_level: v}))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="silent">Silent</SelectItem>
                <SelectItem value="notify">Notify Only</SelectItem>
                <SelectItem value="act_notify">Act & Notify</SelectItem>
                <SelectItem value="autonomous">Fully Autonomous</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Thinking Level */}
          <div className="space-y-2">
            <Label>Thinking Level</Label>
            <Select value={soulSettings.thinking_level} onValueChange={(v) => setSoulSettings(s => ({...s, thinking_level: v}))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="off">Off</SelectItem>
                <SelectItem value="minimal">Minimal</SelectItem>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="high">High</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Save Soul button */}
          <Button onClick={saveSoulSettings} disabled={savingSoul} className="w-full">
            {savingSoul ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
            Save Soul Settings
          </Button>
        </CardContent>
      </Card>

      {/* PRD-55: Orchestrator Heartbeat */}
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" /> Orchestrator Heartbeat</CardTitle>
          <CardDescription>Configure proactive monitoring for your orchestrator</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Enable toggle */}
          <div className="flex items-center justify-between">
            <Label>Enable Heartbeat</Label>
            <Switch checked={heartbeatSettings.enabled} onCheckedChange={(v) => setHeartbeatSettings(s => ({...s, enabled: v}))} />
          </div>

          {heartbeatSettings.enabled && (
            <>
              {/* Interval */}
              <div className="space-y-2">
                <Label>Interval (minutes)</Label>
                <Select value={String(heartbeatSettings.interval_minutes)} onValueChange={(v) => setHeartbeatSettings(s => ({...s, interval_minutes: Number(v)}))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="15">15 minutes</SelectItem>
                    <SelectItem value="30">30 minutes</SelectItem>
                    <SelectItem value="60">1 hour</SelectItem>
                    <SelectItem value="120">2 hours</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Active Hours */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Active From</Label>
                  <Input type="time" value={heartbeatSettings.active_hours_start} onChange={(e) => setHeartbeatSettings(s => ({...s, active_hours_start: e.target.value}))} />
                </div>
                <div className="space-y-2">
                  <Label>Active Until</Label>
                  <Input type="time" value={heartbeatSettings.active_hours_end} onChange={(e) => setHeartbeatSettings(s => ({...s, active_hours_end: e.target.value}))} />
                </div>
              </div>

              {/* Timezone */}
              <div className="space-y-2">
                <Label>Timezone</Label>
                <Input value={heartbeatSettings.timezone} onChange={(e) => setHeartbeatSettings(s => ({...s, timezone: e.target.value}))} placeholder="UTC" />
              </div>

              {/* Heartbeat checklist */}
              <div className="space-y-2">
                <Label>Heartbeat Checklist</Label>
                <textarea className="w-full min-h-[100px] rounded-md border bg-background px-3 py-2 text-sm"
                  value={heartbeatSettings.checklist}
                  onChange={(e) => setHeartbeatSettings(s => ({...s, checklist: e.target.value}))}
                  placeholder={'- Check pending tasks\n- Review unrouted events\n- Summarize today\'s activity'} />
              </div>
            </>
          )}

          <Button onClick={saveHeartbeatSettings} disabled={savingHeartbeat} className="w-full">
            {savingHeartbeat ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
            Save Heartbeat Settings
          </Button>
        </CardContent>
      </Card>

      {/* PRD-55: What Automatos Knows */}
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Brain className="h-5 w-5" /> What Automatos Knows</CardTitle>
          <CardDescription>Memories stored about you and your preferences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {memoryLoading ? (
            <div className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin" /> Loading memories...</div>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{memoryStats.total}</p>
                  <p className="text-xs text-muted-foreground">Total Memories</p>
                </div>
                <div className="text-center p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{memoryStats.agents}</p>
                  <p className="text-xs text-muted-foreground">Agents with Memory</p>
                </div>
                <div className="text-center p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{memoryStats.recent24h}</p>
                  <p className="text-xs text-muted-foreground">Last 24h</p>
                </div>
              </div>

              {memoryStats.recent?.length > 0 && (
                <div className="space-y-2">
                  <Label>Recent Memories</Label>
                  {memoryStats.recent.map((m: any, i: number) => (
                    <div key={i} className="p-2 rounded border bg-muted/30 text-sm truncate">{m.preview || m.memory}</div>
                  ))}
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={handleReset} disabled={saving}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset to Defaults
        </Button>
        <Button onClick={handleSave} disabled={saving}>
          <Save className="h-4 w-4 mr-2" />
          Save Changes
        </Button>
      </div>
    </div>
  )
}
