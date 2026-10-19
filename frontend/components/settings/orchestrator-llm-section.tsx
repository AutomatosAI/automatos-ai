/**
 * Settings > Orchestrator, section 1: LLM Configuration (PRD-55) — provider, model,
 * thinking level, sampling and the advanced performance settings of the
 * orchestrator (the Auto agent). Moved out of SystemLLMSettingsTab.tsx (PRD-263
 * PR-1E-a) so each part fits the component length rule; markup unchanged.
 */

import React, { useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Save, RotateCcw, Brain, Zap, ChevronDown } from 'lucide-react'
import { InlineHelp } from '@/components/ui/help-tooltip'
import { useWorkspaceModels } from '@/hooks/use-model-api'
import { chatProviders, hostsVendorModels, useProviderRegistry } from '@/hooks/use-provider-registry'
import { OrchestratorConfig, THINKING_LEVELS } from './orchestrator-settings-config'

type LLMChange = (key: string, value: string | number | string[] | null) => void

/**
 * The installed models the selected provider serves, and the providers to pick from.
 * The tab calls it at mount, before its loading state, so the catalog loads as early
 * as it did when the tab was one component.
 */
export function useOrchestratorModelOptions(selectedProvider: string) {
  // Load workspace-installed models (same catalog agents use)
  const { data: allModels = [], isLoading: modelsLoading } = useWorkspaceModels()

  // PRD-236: providers come from the backend registry; a provider that hosts
  // vendor-prefixed ids (OpenRouter, NVIDIA) lists the aggregator-tier models.
  const registry = useProviderRegistry()
  const providerOptions = useMemo(() => chatProviders(registry), [registry])

  const availableModels = useMemo(() => {
    if (!Array.isArray(allModels)) return []
    if (!selectedProvider) return allModels
    // PRD-236 W1: an installed row is a ROUTE — list the ones served by the
    // selected provider. Rows from an older API (no serving_provider) keep the
    // vendor/aggregator heuristic.
    const isAggregator = hostsVendorModels(registry, selectedProvider)
    return allModels.filter((model: any) =>
      model.serving_provider
        ? model.serving_provider === selectedProvider
        : model.provider === selectedProvider || (isAggregator && (model.sourcing ?? model.tier) === 'aggregator')
    )
  }, [allModels, selectedProvider, registry])
  return { selectedProvider, providerOptions, availableModels, modelsLoading }
}

type ModelOptions = ReturnType<typeof useOrchestratorModelOptions>

interface OrchestratorLLMSectionProps {
  orchConfig: OrchestratorConfig | null
  modelOptions: ModelOptions
  formData: Record<string, string>
  saving: boolean
  onLLMChange: LLMChange
  onOrchChange: (key: string, value: string | boolean) => void
  onInputChange: (key: string, value: string) => void
  onSave: () => void
  onReset: () => void
}

export function OrchestratorLLMSection(props: OrchestratorLLMSectionProps) {
  const { orchConfig, modelOptions, formData, saving, onLLMChange, onOrchChange, onInputChange, onSave, onReset } = props
  const [llmOpen, setLlmOpen] = useState(true)

  return (
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
                    This workspace&apos;s Auto: provider, model and performance. Reset returns it to the platform default an admin sets.
                  </CardDescription>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 transition-transform ${llmOpen ? 'rotate-180' : ''}`} />
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-6">
            <LLMProviderModelFields orchConfig={orchConfig} modelOptions={modelOptions} onLLMChange={onLLMChange} />
            <LLMTuningFields orchConfig={orchConfig} onLLMChange={onLLMChange} onOrchChange={onOrchChange} />
            <LLMAdvancedSettings
              orchConfig={orchConfig}
              formData={formData}
              onLLMChange={onLLMChange}
              onInputChange={onInputChange}
            />

            {/* LLM Save */}
            <div className="flex justify-end gap-2">
              <Button variant="outline" size="sm" onClick={onReset} disabled={saving}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset
              </Button>
              <Button size="sm" onClick={onSave} disabled={saving}>
                <Save className="h-4 w-4 mr-2" />
                Save LLM Settings
              </Button>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

interface LLMProviderModelFieldsProps {
  orchConfig: OrchestratorConfig | null
  modelOptions: ModelOptions
  onLLMChange: LLMChange
}

function LLMProviderModelFields({ orchConfig, modelOptions, onLLMChange }: LLMProviderModelFieldsProps) {
  const { selectedProvider, providerOptions, availableModels, modelsLoading } = modelOptions

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="space-y-2">
        <Label htmlFor="llm_provider" className="flex items-center gap-1">LLM Provider <InlineHelp id="settings.llm.provider" size="sm" /></Label>
        <Select
          value={orchConfig?.llm?.provider || ''}
          onValueChange={(value) => onLLMChange('provider', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select LLM provider" />
          </SelectTrigger>
          <SelectContent>
            {providerOptions.map((p) => (
              <SelectItem key={p.slug} value={p.slug}>
                {p.label}
                {p.free ? ' (free tier)' : ''}
              </SelectItem>
            ))}
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
            // PRD-236 W1: the list is already the selected provider's routes,
            // so the pick never changes the provider (the tag is the route).
            onLLMChange('model_id', value)
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
              availableModels.map((model: { id?: number; model_id: string; display_name: string; context_window: number; is_free?: boolean }) => (
                <SelectItem key={model.id ?? model.model_id} value={model.model_id}>
                  {model.display_name}{model.is_free ? ' · free' : ''}
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
  )
}

interface LLMTuningFieldsProps {
  orchConfig: OrchestratorConfig | null
  onLLMChange: LLMChange
  onOrchChange: (key: string, value: string | boolean) => void
}

function LLMTuningFields({ orchConfig, onLLMChange, onOrchChange }: LLMTuningFieldsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="space-y-2">
        <Label className="flex items-center gap-1">Thinking Level <InlineHelp id="settings.llm.thinking_level" size="sm" /></Label>
        <Select
          value={orchConfig?.thinking_level || 'medium'}
          onValueChange={(v) => onOrchChange('thinking_level', v)}
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
          onChange={(e) => onLLMChange('temperature', parseFloat(e.target.value) || 0)}
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
          onChange={(e) => onLLMChange('max_tokens', parseInt(e.target.value) || 0)}
          placeholder="4000"
        />
        <p className="text-xs text-muted-foreground">Max tokens in response</p>
      </div>
    </div>
  )
}

interface LLMAdvancedSettingsProps {
  orchConfig: OrchestratorConfig | null
  formData: Record<string, string>
  onLLMChange: LLMChange
  onInputChange: (key: string, value: string) => void
}

function LLMAdvancedSettings({ orchConfig, formData, onLLMChange, onInputChange }: LLMAdvancedSettingsProps) {
  return (
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
              onChange={(e) => onLLMChange('timeout', parseInt(e.target.value) || null)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Retry Attempts <InlineHelp id="settings.llm.retry_attempts" size="sm" /></Label>
            <Input
              type="number" min="0" max="5"
              value={formData.llm_retry_attempts || '3'}
              onChange={(e) => onInputChange('llm_retry_attempts', e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Concurrent Requests <InlineHelp id="settings.llm.concurrent_requests" size="sm" /></Label>
            <Input
              type="number" min="1" max="10"
              value={formData.llm_concurrent_requests || '5'}
              onChange={(e) => onInputChange('llm_concurrent_requests', e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Cache TTL (seconds) <InlineHelp id="settings.llm.cache_ttl" size="sm" /></Label>
            <Input
              type="number" min="0" max="3600"
              value={formData.llm_cache_ttl || '300'}
              onChange={(e) => onInputChange('llm_cache_ttl', e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Top P <InlineHelp id="settings.llm.top_p" size="sm" /></Label>
            <Input
              type="number" step="0.1" min="0" max="1"
              value={orchConfig?.llm?.top_p ?? 1}
              onChange={(e) => onLLMChange('top_p', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Frequency Penalty <InlineHelp id="settings.llm.frequency_penalty" size="sm" /></Label>
            <Input
              type="number" step="0.1" min="-2" max="2"
              value={orchConfig?.llm?.frequency_penalty ?? 0}
              onChange={(e) => onLLMChange('frequency_penalty', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Presence Penalty <InlineHelp id="settings.llm.presence_penalty" size="sm" /></Label>
            <Input
              type="number" step="0.1" min="-2" max="2"
              value={orchConfig?.llm?.presence_penalty ?? 0}
              onChange={(e) => onLLMChange('presence_penalty', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="space-y-2">
            <Label className="flex items-center gap-1">Stop Sequences <InlineHelp id="settings.llm.stop_sequences" size="sm" /></Label>
            <Input
              value={(orchConfig?.llm?.stop || []).join(', ')}
              onChange={(e) => {
                const val = e.target.value
                const sequences = val ? val.split(',').map((s: string) => s.trim()).filter(Boolean) : null
                onLLMChange('stop', sequences)
              }}
              placeholder="\n\n, ###, END"
            />
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
