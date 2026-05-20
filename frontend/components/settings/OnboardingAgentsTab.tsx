'use client'

import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Save, ChevronDown, ChevronUp, RefreshCw, Rocket, Pencil, Brain, Target, ShieldCheck } from 'lucide-react'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import {
  getSettingsByCategory,
  bulkUpdateSettings,
  SystemSettingsByCategory,
} from '@/lib/api/system-settings'

interface OnboardingAgent {
  id: number
  slug: string
  name: string
  description: string
  status: string
  job_title: string | null
  team: string | null
  model_id: string
  provider: string
  temperature: number
  max_tokens: number
  custom_persona_prompt: string
  tags: string[]
  configuration: Record<string, unknown>
}

const MODEL_TIERS: Record<string, { label: string; color: string }> = {
  'anthropic/claude-opus-4-20250514': { label: 'Premium', color: 'bg-agent' },
  'anthropic/claude-sonnet-4-20250514': { label: 'Mid', color: 'bg-blue-500' },
  'openai/gpt-4.1': { label: 'Mid', color: 'bg-green-500' },
  'deepseek/deepseek-chat': { label: 'Budget', color: 'bg-yellow-500' },
}

// Keys we show in the Planner card (from coordination category)
const PLANNER_KEYS = ['provider', 'model', 'planner_max_tokens', 'planner_temperature'] as const
// Keys we show in the Verifier card (from coordination category)
const VERIFIER_KEYS = ['verifier_fallback_model', 'verifier_model_mapping', 'verifier_max_tokens', 'verification_pass_threshold', 'verification_catastrophic_threshold'] as const
// All coordination keys we manage
const ALL_COORD_KEYS = [...PLANNER_KEYS, ...VERIFIER_KEYS] as const

export function OnboardingAgentsTab() {
  const [agents, setAgents] = useState<OnboardingAgent[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState<string | null>(null)
  const [expandedSlug, setExpandedSlug] = useState<string | null>(null)
  const [editState, setEditState] = useState<Record<string, Partial<OnboardingAgent>>>({})

  // Planner settings from system_settings (coordination category)
  const [plannerSettings, setPlannerSettings] = useState<SystemSettingsByCategory | null>(null)
  const [plannerValues, setPlannerValues] = useState<Record<string, string>>({})
  const [plannerDirty, setPlannerDirty] = useState(false)
  const [plannerSaving, setPlannerSaving] = useState(false)

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true)
      const [agentsData, allSettings] = await Promise.all([
        apiClient.request<{ agents: OnboardingAgent[] }>('/api/settings/onboarding-agents'),
        getSettingsByCategory(),
      ])
      setAgents(agentsData.agents || [])

      // Extract coordination category for planner + verifier cards
      const coord = allSettings.find(c => c.category === 'coordination')
      if (coord) {
        setPlannerSettings(coord)
        const vals: Record<string, string> = {}
        for (const s of coord.settings) {
          if ((ALL_COORD_KEYS as readonly string[]).includes(s.key)) {
            vals[s.key] = s.value ?? s.default_value ?? ''
          }
        }
        setPlannerValues(vals)
        setPlannerDirty(false)
      }
    } catch (err) {
      console.error('Failed to load onboarding agents:', err)
      toast.error('Failed to load onboarding agents')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAgents()
  }, [fetchAgents])

  const getEdits = (slug: string) => editState[slug] || {}

  const updateEdit = (slug: string, field: string, value: unknown) => {
    setEditState(prev => ({
      ...prev,
      [slug]: { ...prev[slug], [field]: value },
    }))
  }

  const handleSave = async (slug: string) => {
    const edits = getEdits(slug)
    if (Object.keys(edits).length === 0) return

    try {
      setSaving(slug)
      await apiClient.request(`/api/settings/onboarding-agents/${slug}`, {
        method: 'PUT',
        body: JSON.stringify(edits),
      })
      toast.success(`${slug.replace('onboarding-', '').toUpperCase()} updated`)
      setEditState(prev => {
        const next = { ...prev }
        delete next[slug]
        return next
      })
      await fetchAgents()
    } catch (err) {
      console.error('Failed to update agent:', err)
      toast.error('Failed to update agent')
    } finally {
      setSaving(null)
    }
  }

  const handleToggleStatus = async (agent: OnboardingAgent) => {
    const newStatus = agent.status === 'active' ? 'inactive' : 'active'
    try {
      await apiClient.request(`/api/settings/onboarding-agents/${agent.slug}`, {
        method: 'PUT',
        body: JSON.stringify({ status: newStatus }),
      })
      toast.success(`${agent.name} ${newStatus === 'active' ? 'enabled' : 'disabled'}`)
      await fetchAgents()
    } catch (err) {
      toast.error('Failed to toggle agent status')
    }
  }

  const handlePlannerChange = (key: string, val: string) => {
    setPlannerValues(prev => ({ ...prev, [key]: val }))
    setPlannerDirty(true)
  }

  const handleCoordinationSave = async () => {
    if (!plannerSettings) return
    try {
      setPlannerSaving(true)
      const updates = plannerSettings.settings
        .filter(s => (ALL_COORD_KEYS as readonly string[]).includes(s.key))
        .filter(s => {
          const current = plannerValues[s.key] ?? ''
          const original = s.value ?? s.default_value ?? ''
          return current !== original
        })
        .map(s => ({ id: s.id, value: plannerValues[s.key] ?? '' }))

      if (updates.length > 0) {
        await bulkUpdateSettings(updates)
        toast.success('Mission pipeline settings saved')
        setPlannerDirty(false)
        await fetchAgents()
      }
    } catch (err) {
      console.error('Failed to save coordination settings:', err)
      toast.error('Failed to save settings')
    } finally {
      setPlannerSaving(false)
    }
  }

  const tierInfo = (modelId: string) => MODEL_TIERS[modelId] || { label: 'Custom', color: 'bg-gray-500' }

  if (loading) {
    return (
      <Card>
        <CardContent className="py-10 text-center text-muted-foreground">
          Loading onboarding agents...
        </CardContent>
      </Card>
    )
  }

  if (agents.length === 0) {
    return (
      <Card>
        <CardContent className="py-10 text-center text-muted-foreground">
          No onboarding agents found. They will be seeded on next API restart.
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            Onboarding Agents (Mission Zero)
          </CardTitle>
          <CardDescription>
            Hidden system agents that power the new-workspace onboarding experience.
            They research the business, design the agent roster, write personas, and build the workspace.
            Use high-quality models here to create a premium first impression during free trials.
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Mission Planner LLM Settings */}
      {plannerSettings && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Target className="h-5 w-5 text-primary" />
                <div>
                  <CardTitle className="text-base">Mission Planner</CardTitle>
                  <CardDescription>
                    Decomposes the mission into tasks and assigns them to agents. This is the first LLM call in every mission.
                  </CardDescription>
                </div>
              </div>
              <Button
                size="sm"
                disabled={!plannerDirty || plannerSaving}
                onClick={handleCoordinationSave}
              >
                <Save className="h-4 w-4 mr-2" />
                {plannerSaving ? 'Saving...' : 'Save Pipeline'}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label>Provider</Label>
                <Input
                  value={plannerValues.provider ?? ''}
                  onChange={(e) => handlePlannerChange('provider', e.target.value)}
                  placeholder="openrouter"
                />
              </div>
              <div className="space-y-2">
                <Label>Model</Label>
                <Input
                  value={plannerValues.model ?? ''}
                  onChange={(e) => handlePlannerChange('model', e.target.value)}
                  placeholder="openai/gpt-4o-mini"
                />
              </div>
              <div className="space-y-2">
                <Label>Max Tokens</Label>
                <Input
                  type="number"
                  min={500}
                  max={32000}
                  value={plannerValues.planner_max_tokens ?? ''}
                  onChange={(e) => handlePlannerChange('planner_max_tokens', e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Temperature</Label>
                <Input
                  type="number"
                  step={0.1}
                  min={0}
                  max={1}
                  value={plannerValues.planner_temperature ?? ''}
                  onChange={(e) => handlePlannerChange('planner_temperature', e.target.value)}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Verification Settings */}
      {plannerSettings && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <ShieldCheck className="h-5 w-5 text-success" />
              <div>
                <CardTitle className="text-base">Verification (Advisory)</CardTitle>
                <CardDescription>
                  Reviews task outputs after completion. Advisory only — never rejects unless catastrophically low score.
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Fallback Model</Label>
                <Input
                  value={plannerValues.verifier_fallback_model ?? ''}
                  onChange={(e) => handlePlannerChange('verifier_fallback_model', e.target.value)}
                  placeholder="openai/gpt-4o-mini"
                />
              </div>
              <div className="space-y-2">
                <Label>Pass Threshold</Label>
                <Input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={plannerValues.verification_pass_threshold ?? ''}
                  onChange={(e) => handlePlannerChange('verification_pass_threshold', e.target.value)}
                  placeholder="0.7"
                />
                <p className="text-xs text-muted-foreground">Score above this = pass</p>
              </div>
              <div className="space-y-2">
                <Label>Catastrophic Threshold</Label>
                <Input
                  type="number"
                  step={0.05}
                  min={0}
                  max={0.5}
                  value={plannerValues.verification_catastrophic_threshold ?? ''}
                  onChange={(e) => handlePlannerChange('verification_catastrophic_threshold', e.target.value)}
                  placeholder="0.15"
                />
                <p className="text-xs text-muted-foreground">Below this = flag for review</p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Max Tokens</Label>
                <Input
                  type="number"
                  min={200}
                  max={8000}
                  value={plannerValues.verifier_max_tokens ?? ''}
                  onChange={(e) => handlePlannerChange('verifier_max_tokens', e.target.value)}
                  placeholder="2000"
                />
              </div>
              <div className="space-y-2">
                <Label>Model Mapping</Label>
                <Input
                  value={plannerValues.verifier_model_mapping ?? ''}
                  onChange={(e) => handlePlannerChange('verifier_model_mapping', e.target.value)}
                  placeholder="anthropic=openai/gpt-4o-mini,openai=anthropic/claude-haiku-4-5"
                />
                <p className="text-xs text-muted-foreground">Cross-model verification: family=model pairs</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {agents.map((agent) => {
        const edits = getEdits(agent.slug)
        const isExpanded = expandedSlug === agent.slug
        const hasEdits = Object.keys(edits).length > 0
        const tier = tierInfo(edits.model_id || agent.model_id)
        const agentName = agent.slug.replace('onboarding-', '').toUpperCase()

        return (
          <Card key={agent.slug} className={agent.status === 'inactive' ? 'opacity-60' : ''}>
            <CardHeader
              className="cursor-pointer"
              onClick={() => setExpandedSlug(isExpanded ? null : agent.slug)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Brain className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      {agentName}
                      <Badge variant="outline" className="text-xs">
                        {agent.job_title}
                      </Badge>
                      <Badge className={`text-xs text-white ${tier.color}`}>
                        {tier.label}
                      </Badge>
                    </CardTitle>
                    <CardDescription className="mt-0.5">
                      {agent.description}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Switch
                    checked={agent.status === 'active'}
                    onCheckedChange={() => handleToggleStatus(agent)}
                    onClick={(e) => e.stopPropagation()}
                  />
                  {isExpanded ? (
                    <ChevronUp className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>
              </div>
            </CardHeader>

            {isExpanded && (
              <CardContent className="space-y-4 pt-0">
                {/* Model Configuration */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label>Model ID</Label>
                    <Input
                      value={edits.model_id ?? agent.model_id}
                      onChange={(e) => updateEdit(agent.slug, 'model_id', e.target.value)}
                      placeholder="anthropic/claude-opus-4-20250514"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Temperature</Label>
                    <Input
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={edits.temperature ?? agent.temperature}
                      onChange={(e) => updateEdit(agent.slug, 'temperature', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Max Tokens</Label>
                    <Input
                      type="number"
                      min="1000"
                      max="32000"
                      value={edits.max_tokens ?? agent.max_tokens}
                      onChange={(e) => updateEdit(agent.slug, 'max_tokens', parseInt(e.target.value) || 8000)}
                    />
                  </div>
                </div>

                {/* Persona */}
                <div className="space-y-2">
                  <Label className="flex items-center gap-2">
                    <Pencil className="h-3.5 w-3.5" />
                    System Prompt (Persona)
                  </Label>
                  <Textarea
                    value={edits.custom_persona_prompt ?? agent.custom_persona_prompt}
                    onChange={(e) => updateEdit(agent.slug, 'custom_persona_prompt', e.target.value)}
                    rows={12}
                    className="font-mono text-sm"
                  />
                </div>

                {/* Save */}
                <div className="flex justify-end gap-2">
                  {hasEdits && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setEditState(prev => {
                        const next = { ...prev }
                        delete next[agent.slug]
                        return next
                      })}
                    >
                      Discard
                    </Button>
                  )}
                  <Button
                    size="sm"
                    disabled={!hasEdits || saving === agent.slug}
                    onClick={() => handleSave(agent.slug)}
                  >
                    <Save className="h-4 w-4 mr-2" />
                    {saving === agent.slug ? 'Saving...' : 'Save'}
                  </Button>
                </div>
              </CardContent>
            )}
          </Card>
        )
      })}

      {/* Refresh */}
      <div className="flex justify-end">
        <Button variant="ghost" size="sm" onClick={fetchAgents}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>
    </div>
  )
}
