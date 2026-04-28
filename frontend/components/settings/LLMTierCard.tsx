/**
 * LLM Tier Card (PRD-136)
 * =======================
 *
 * Generic, canonical-schema card for one of the 3 LLM tiers:
 *   - orchestrator_llm  (Auto — the brain)
 *   - system_llm        (background workers — codegraph, RAG, memory, etc.)
 *   - embeddings        (RAG/memory/search vectors)
 *
 * Renders fields directly from system_settings rows. Tooltips come from
 * each row's `description`. Field kind is driven by `value_type`:
 *   string  → <Input type="text"> (or <Select> when key is "provider"/"model")
 *   integer → <Input type="number">
 *   float   → <Input type="number" step="0.1">
 *   boolean → <Switch>
 *
 * Only one shape exists. No per-tier custom logic — the seed defines
 * which keys live in basic vs advanced via TIER_FIELD_GROUPS below.
 */

import React, { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChevronDown, Save, RotateCcw, Loader2 } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { HelpCircle } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

const LLM_BASIC = ['provider', 'model', 'temperature', 'max_tokens'] as const
const EMBEDDINGS_BASIC = ['provider', 'model', 'dimensions', 'chunk_size'] as const

const TIER_FIELD_GROUPS: Record<string, { basic: string[]; order: string[] }> = {
  orchestrator_llm: {
    basic: [...LLM_BASIC],
    order: ['provider', 'model', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'timeout_seconds', 'max_retries'],
  },
  system_llm: {
    basic: [...LLM_BASIC],
    order: ['provider', 'model', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'timeout_seconds', 'max_retries'],
  },
  embeddings: {
    basic: [...EMBEDDINGS_BASIC],
    order: ['provider', 'model', 'dimensions', 'chunk_size', 'chunk_overlap', 'max_seq_length', 'vector_store_type', 'cache_dir', 'rerank_enabled', 'rerank_model'],
  },
}

const PROVIDER_OPTIONS: Record<string, string[]> = {
  llm: ['openai', 'anthropic', 'google', 'openrouter', 'deepseek', 'azure', 'bedrock', 'grok', 'cohere', 'huggingface', 'local'],
  embeddings: ['openrouter', 'openai', 'google', 'cohere', 'huggingface_local', 'huggingface_api', 'disabled'],
}

const VECTOR_STORE_OPTIONS = ['pgvector', 'faiss', 'chroma', 'pinecone', 'weaviate']

interface LLMTierCardProps {
  category: 'orchestrator_llm' | 'system_llm' | 'embeddings'
  title: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => Promise<void> | void
  onReset: () => Promise<void> | void
  saving: boolean
}

function FieldTooltip({ description }: { description: string | null }) {
  if (!description) return null
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground cursor-help" />
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <p className="text-xs">{description}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

function humanizeKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function renderField(
  setting: SystemSetting,
  category: string,
  value: string,
  onChange: (next: string) => void,
) {
  const id = `${category}_${setting.key}`
  const label = humanizeKey(setting.key)
  const isProvider = setting.key === 'provider'
  const isModel = setting.key === 'model'
  const isVectorStore = setting.key === 'vector_store_type'
  const isBool = setting.value_type === 'boolean' || setting.value_type === 'bool'
  const isNumeric = setting.value_type === 'integer' || setting.value_type === 'float' || setting.value_type === 'int' || setting.value_type === 'number'
  const isFloat = setting.value_type === 'float'

  const labelEl = (
    <Label htmlFor={id} className="flex items-center gap-1.5">
      {label}
      <FieldTooltip description={setting.description} />
    </Label>
  )

  if (isProvider) {
    const options = category === 'embeddings' ? PROVIDER_OPTIONS.embeddings : PROVIDER_OPTIONS.llm
    return (
      <div className="space-y-2" key={id}>
        {labelEl}
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger id={id}><SelectValue placeholder="Select provider" /></SelectTrigger>
          <SelectContent>
            {options.map((opt) => (
              <SelectItem key={opt} value={opt}>{opt}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }

  if (isVectorStore) {
    return (
      <div className="space-y-2" key={id}>
        {labelEl}
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger id={id}><SelectValue placeholder="Select vector store" /></SelectTrigger>
          <SelectContent>
            {VECTOR_STORE_OPTIONS.map((opt) => (
              <SelectItem key={opt} value={opt}>{opt}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }

  if (isBool) {
    return (
      <div className="space-y-2" key={id}>
        {labelEl}
        <div className="flex items-center gap-2">
          <Switch
            id={id}
            checked={value === 'true'}
            onCheckedChange={(checked) => onChange(checked ? 'true' : 'false')}
          />
          <span className="text-sm text-muted-foreground">{value === 'true' ? 'Enabled' : 'Disabled'}</span>
        </div>
      </div>
    )
  }

  if (isModel || setting.key === 'rerank_model' || setting.key === 'cache_dir') {
    return (
      <div className="space-y-2" key={id}>
        {labelEl}
        <Input
          id={id}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={setting.default_value || ''}
        />
      </div>
    )
  }

  return (
    <div className="space-y-2" key={id}>
      {labelEl}
      <Input
        id={id}
        type={isNumeric ? 'number' : 'text'}
        step={isFloat ? '0.1' : undefined}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={setting.default_value || ''}
      />
    </div>
  )
}

export default function LLMTierCard({
  category,
  title,
  description,
  icon: Icon,
  settings,
  onSave,
  onReset,
  saving,
}: LLMTierCardProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})
  const [advancedOpen, setAdvancedOpen] = useState(false)

  useEffect(() => {
    const initial: Record<string, string> = {}
    settings.forEach((s) => {
      initial[s.key] = s.value !== null && s.value !== undefined ? s.value : (s.default_value || '')
    })
    setFormData(initial)
  }, [settings])

  const config = TIER_FIELD_GROUPS[category]

  const { basicSettings, advancedSettings } = useMemo(() => {
    const byKey = new Map(settings.map((s) => [s.key, s]))
    const basic: SystemSetting[] = []
    const advanced: SystemSetting[] = []
    const ordered = config.order
      .map((k) => byKey.get(k))
      .filter((s): s is SystemSetting => Boolean(s))
    // Append any settings not in the explicit order list (forward-compat for new keys)
    const knownKeys = new Set(config.order)
    settings.forEach((s) => {
      if (!knownKeys.has(s.key)) ordered.push(s)
    })
    ordered.forEach((s) => {
      if (config.basic.includes(s.key)) basic.push(s)
      else advanced.push(s)
    })
    return { basicSettings: basic, advancedSettings: advanced }
  }, [settings, config])

  const handleChange = (key: string, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    onSave(formData)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className="h-5 w-5" />
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Basic fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {basicSettings.map((s) =>
            renderField(s, category, formData[s.key] ?? '', (v) => handleChange(s.key, v)),
          )}
        </div>

        {/* Advanced (collapsible) */}
        {advancedSettings.length > 0 && (
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="gap-2">
                Advanced Settings
                <ChevronDown className={`h-3 w-3 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {advancedSettings.map((s) =>
                  renderField(s, category, formData[s.key] ?? '', (v) => handleChange(s.key, v)),
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-2 border-t border-border/30">
          <Button variant="outline" size="sm" onClick={() => onReset()} disabled={saving}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button size="sm" onClick={handleSave} disabled={saving}>
            {saving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Save className="h-4 w-4 mr-2" />}
            Save
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
