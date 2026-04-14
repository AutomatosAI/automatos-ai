/**
 * System LLMs Settings Tab Component
 * ===================================
 *
 * Manages all background/infrastructure LLM configurations:
 * - CodeGraph (code analysis)
 * - Knowledge Graph (entity extraction)
 * - Support Agent (future)
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw, Code, Settings, Zap, Brain, Database, Network } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { SystemSetting } from '@/lib/api/system-settings'

interface CodeGraphSettingsTabProps {
  codegraphSettings: SystemSetting[]
  knowledgeGraphSettings: SystemSetting[]
  onSaveCodegraph: (updates: Record<string, string>) => void
  onSaveKnowledgeGraph: (updates: Record<string, string>) => void
  saving: boolean
  onResetCodegraph: () => void
  onResetKnowledgeGraph: () => void
}

export default function CodeGraphSettingsTab({
  codegraphSettings,
  knowledgeGraphSettings,
  onSaveCodegraph,
  onSaveKnowledgeGraph,
  saving,
  onResetCodegraph,
  onResetKnowledgeGraph,
}: CodeGraphSettingsTabProps) {
  const [cgForm, setCgForm] = useState<Record<string, string>>({})
  const [kgForm, setKgForm] = useState<Record<string, string>>({})

  React.useEffect(() => {
    const data: Record<string, string> = {}
    codegraphSettings.forEach(s => {
      data[s.key] = s.value !== null && s.value !== undefined ? s.value : (s.default_value || '')
    })
    setCgForm(data)
  }, [codegraphSettings])

  React.useEffect(() => {
    const data: Record<string, string> = {}
    knowledgeGraphSettings.forEach(s => {
      data[s.key] = s.value !== null && s.value !== undefined ? s.value : (s.default_value || '')
    })
    setKgForm(data)
  }, [knowledgeGraphSettings])

  const cgChange = (key: string, value: string) => setCgForm(prev => ({ ...prev, [key]: value }))
  const kgChange = (key: string, value: string) => setKgForm(prev => ({ ...prev, [key]: value }))

  const getCgSetting = (key: string) => codegraphSettings.find(s => s.key === key)

  return (
    <div className="space-y-6">
      {/* ── CodeGraph Section ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            CodeGraph LLM
          </CardTitle>
          <CardDescription>
            LLM provider and model for code analysis and graph generation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>LLM Provider</Label>
              <Select value={cgForm.provider || ''} onValueChange={(v) => cgChange('provider', v)}>
                <SelectTrigger><SelectValue placeholder="Select provider" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="google">Google</SelectItem>
                  <SelectItem value="openrouter">OpenRouter</SelectItem>
                  <SelectItem value="azure">Azure OpenAI</SelectItem>
                  <SelectItem value="huggingface">HuggingFace</SelectItem>
                </SelectContent>
              </Select>
              {getCgSetting('provider')?.is_required && <Badge variant="destructive" className="text-xs">Required</Badge>}
            </div>
            <div className="space-y-2">
              <Label>LLM Model</Label>
              <Input
                value={cgForm.model || ''}
                onChange={(e) => cgChange('model', e.target.value)}
                placeholder="gpt-4o-mini"
              />
              {getCgSetting('model')?.is_required && <Badge variant="destructive" className="text-xs">Required</Badge>}
            </div>
          </div>

          {/* Embedding Model */}
          <div className="space-y-2">
            <Label>Embedding Model</Label>
            <Select value={cgForm.embedding_model || ''} onValueChange={(v) => cgChange('embedding_model', v)}>
              <SelectTrigger><SelectValue placeholder="Select embedding model" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="text-embedding-3-small">OpenAI Embedding 3 Small (1536d)</SelectItem>
                <SelectItem value="text-embedding-3-large">OpenAI Embedding 3 Large (3072d)</SelectItem>
                <SelectItem value="text-embedding-ada-002">OpenAI Ada-002 (1536d)</SelectItem>
                <SelectItem value="sentence-transformers/all-MiniLM-L6-v2">MiniLM (384d, Free)</SelectItem>
                <SelectItem value="sentence-transformers/all-mpnet-base-v2">MPNet (768d, Free)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* CodeGraph Config */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            CodeGraph Configuration
          </CardTitle>
          <CardDescription>Code analysis, graph generation, and performance settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Enable CodeGraph Analysis</Label>
              <p className="text-sm text-muted-foreground">Automatic code analysis and graph generation</p>
            </div>
            <Switch
              checked={cgForm.enabled === 'true'}
              onCheckedChange={(checked) => cgChange('enabled', checked.toString())}
            />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Max File Size (bytes)</Label>
              <Input type="number" value={cgForm.max_file_size || ''} onChange={(e) => cgChange('max_file_size', e.target.value)} placeholder="1000000" />
            </div>
            <div className="space-y-2">
              <Label>Cache TTL (sec)</Label>
              <Input type="number" value={cgForm.cache_ttl || ''} onChange={(e) => cgChange('cache_ttl', e.target.value)} placeholder="3600" />
            </div>
            <div className="space-y-2">
              <Label>Max Depth</Label>
              <Input type="number" min={1} max={10} value={cgForm.max_depth || '5'} onChange={(e) => cgChange('max_depth', e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Max Nodes</Label>
              <Input type="number" min={100} max={10000} value={cgForm.max_nodes || '1000'} onChange={(e) => cgChange('max_nodes', e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Timeout (sec)</Label>
              <Input type="number" min={10} max={300} value={cgForm.analysis_timeout || '60'} onChange={(e) => cgChange('analysis_timeout', e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Concurrent</Label>
              <Input type="number" min={1} max={10} value={cgForm.concurrent_analyses || '3'} onChange={(e) => cgChange('concurrent_analyses', e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Batch Size</Label>
              <Input type="number" min={1} max={100} value={cgForm.batch_size || '10'} onChange={(e) => cgChange('batch_size', e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Languages</Label>
              <Input value={cgForm.supported_languages || ''} onChange={(e) => cgChange('supported_languages', e.target.value)} placeholder="python,typescript" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* CodeGraph Save */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={() => {
          const defaults: Record<string, string> = {}
          codegraphSettings.forEach(s => { defaults[s.key] = s.default_value || '' })
          setCgForm(defaults)
          onResetCodegraph()
        }} disabled={saving}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset CodeGraph
        </Button>
        <Button onClick={() => onSaveCodegraph(cgForm)} disabled={saving}>
          <Save className="h-4 w-4 mr-2" />
          Save CodeGraph
        </Button>
      </div>

      {/* ── Knowledge Graph Section ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Knowledge Graph LLM
          </CardTitle>
          <CardDescription>
            LLM for entity/relation extraction from documents into the knowledge graph
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Provider</Label>
              <Input
                value={kgForm.provider || ''}
                onChange={(e) => kgChange('provider', e.target.value)}
                placeholder="openrouter"
              />
            </div>
            <div className="space-y-2">
              <Label>Model</Label>
              <Input
                value={kgForm.model || ''}
                onChange={(e) => kgChange('model', e.target.value)}
                placeholder="google/gemini-2.0-flash"
              />
            </div>
            <div className="space-y-2">
              <Label>Max Tokens</Label>
              <Input
                type="number"
                min={500}
                max={8000}
                value={kgForm.extraction_max_tokens || ''}
                onChange={(e) => kgChange('extraction_max_tokens', e.target.value)}
                placeholder="4000"
              />
            </div>
            <div className="space-y-2">
              <Label>Temperature</Label>
              <Input
                type="number"
                step={0.1}
                min={0}
                max={1}
                value={kgForm.extraction_temperature || ''}
                onChange={(e) => kgChange('extraction_temperature', e.target.value)}
                placeholder="0.1"
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label>Max Concurrent Extractions</Label>
            <Input
              type="number"
              min={1}
              max={20}
              value={kgForm.max_concurrent_extractions || ''}
              onChange={(e) => kgChange('max_concurrent_extractions', e.target.value)}
              placeholder="5"
              className="w-32"
            />
          </div>
        </CardContent>
      </Card>

      {/* Knowledge Graph Save */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={() => {
          const defaults: Record<string, string> = {}
          knowledgeGraphSettings.forEach(s => { defaults[s.key] = s.default_value || '' })
          setKgForm(defaults)
          onResetKnowledgeGraph()
        }} disabled={saving}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset Knowledge Graph
        </Button>
        <Button onClick={() => onSaveKnowledgeGraph(kgForm)} disabled={saving}>
          <Save className="h-4 w-4 mr-2" />
          Save Knowledge Graph
        </Button>
      </div>

      {/* ── Support Agent Placeholder ── */}
      <Card className="opacity-60">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Support Agent LLM
          </CardTitle>
          <CardDescription>
            Coming soon — LLM configuration for the customer support agent
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  )
}
