/**
 * CodeGraph Settings Tab Component
 * ================================
 * 
 * Manages CodeGraph analysis settings.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw, Code, Settings, Zap, Brain, Database } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { SystemSetting } from '@/lib/api/system-settings'

interface CodeGraphSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function CodeGraphSettingsTab({ 
  settings, 
  onSave, 
  saving, 
  onReset 
}: CodeGraphSettingsTabProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})

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
      {/* LLM Provider Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            LLM Provider Configuration
          </CardTitle>
          <CardDescription>
            Configure the LLM provider and model for CodeGraph analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="provider">LLM Provider</Label>
              <Select 
                value={formData.provider || ''} 
                onValueChange={(value) => handleInputChange('provider', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select LLM provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="google">Google</SelectItem>
                  <SelectItem value="azure">Azure OpenAI</SelectItem>
                  <SelectItem value="huggingface">HuggingFace</SelectItem>
                </SelectContent>
              </Select>
              {getSetting('provider')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="model">LLM Model</Label>
              <Select 
                value={formData.model || ''} 
                onValueChange={(value) => handleInputChange('model', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4">GPT-4</SelectItem>
                  <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                  <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                  <SelectItem value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</SelectItem>
                  <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                  <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                  <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
                  <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                  <SelectItem value="gemini-pro-vision">Gemini Pro Vision</SelectItem>
                  <SelectItem value="mistralai/Mistral-7B-Instruct-v0.2">Mistral 7B Instruct</SelectItem>
                </SelectContent>
              </Select>
              {getSetting('model')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Embedding Model Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Embedding Model Configuration
          </CardTitle>
          <CardDescription>
            Configure embedding model for CodeGraph semantic search
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="embedding_model">Embedding Model</Label>
            <Select 
              value={formData.embedding_model || ''} 
              onValueChange={(value) => handleInputChange('embedding_model', value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select embedding model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text-embedding-ada-002">OpenAI Ada-002 (1536 dims)</SelectItem>
                <SelectItem value="text-embedding-3-small">OpenAI Embedding 3 Small (1536 dims)</SelectItem>
                <SelectItem value="text-embedding-3-large">OpenAI Embedding 3 Large (3072 dims)</SelectItem>
                <SelectItem value="sentence-transformers/all-MiniLM-L6-v2">HuggingFace MiniLM (384 dims, Free)</SelectItem>
                <SelectItem value="sentence-transformers/all-mpnet-base-v2">HuggingFace MPNet (768 dims, Free)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Note: CodeGraph currently uses OpenAI embeddings. Support for HuggingFace embeddings coming soon.
            </p>
            {getSetting('embedding_model')?.is_required && (
              <Badge variant="destructive" className="text-xs">Required</Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* CodeGraph Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            CodeGraph Configuration
          </CardTitle>
          <CardDescription>
            Configure code analysis and graph generation settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable CodeGraph Analysis</Label>
                <p className="text-sm text-muted-foreground">
                  Enable automatic code analysis and graph generation
                </p>
              </div>
              <Switch
                checked={formData.enabled === 'true'}
                onCheckedChange={(checked) => handleInputChange('enabled', checked.toString())}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="max_file_size">Max File Size (bytes)</Label>
                <Input
                  id="max_file_size"
                  type="number"
                  value={formData.max_file_size || ''}
                  onChange={(e) => handleInputChange('max_file_size', e.target.value)}
                  placeholder="1000000"
                />
                <p className="text-xs text-muted-foreground">
                  Maximum file size for analysis (1MB = 1000000 bytes)
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="cache_ttl">Cache TTL (seconds)</Label>
                <Input
                  id="cache_ttl"
                  type="number"
                  value={formData.cache_ttl || ''}
                  onChange={(e) => handleInputChange('cache_ttl', e.target.value)}
                  placeholder="3600"
                />
                <p className="text-xs text-muted-foreground">
                  How long to cache analysis results
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="supported_languages">Supported Languages</Label>
              <Input
                id="supported_languages"
                value={formData.supported_languages || ''}
                onChange={(e) => handleInputChange('supported_languages', e.target.value)}
                placeholder="python,typescript,javascript,go,rust"
              />
              <p className="text-xs text-muted-foreground">
                Comma-separated list of supported programming languages
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Analysis Settings
          </CardTitle>
          <CardDescription>
            Fine-tune code analysis behavior
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="max_depth">Max Analysis Depth</Label>
              <Input
                id="max_depth"
                type="number"
                min="1"
                max="10"
                value={formData.max_depth || '5'}
                onChange={(e) => handleInputChange('max_depth', e.target.value)}
                placeholder="5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max_nodes">Max Graph Nodes</Label>
              <Input
                id="max_nodes"
                type="number"
                min="100"
                max="10000"
                value={formData.max_nodes || '1000'}
                onChange={(e) => handleInputChange('max_nodes', e.target.value)}
                placeholder="1000"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="analysis_timeout">Analysis Timeout (seconds)</Label>
              <Input
                id="analysis_timeout"
                type="number"
                min="10"
                max="300"
                value={formData.analysis_timeout || '60'}
                onChange={(e) => handleInputChange('analysis_timeout', e.target.value)}
                placeholder="60"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="concurrent_analyses">Concurrent Analyses</Label>
              <Input
                id="concurrent_analyses"
                type="number"
                min="1"
                max="10"
                value={formData.concurrent_analyses || '3'}
                onChange={(e) => handleInputChange('concurrent_analyses', e.target.value)}
                placeholder="3"
              />
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
            Optimize CodeGraph performance and resource usage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="memory_limit">Memory Limit (MB)</Label>
              <Input
                id="memory_limit"
                type="number"
                min="100"
                max="2048"
                value={formData.memory_limit || '512'}
                onChange={(e) => handleInputChange('memory_limit', e.target.value)}
                placeholder="512"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="cpu_limit">CPU Limit (%)</Label>
              <Input
                id="cpu_limit"
                type="number"
                min="10"
                max="100"
                value={formData.cpu_limit || '50'}
                onChange={(e) => handleInputChange('cpu_limit', e.target.value)}
                placeholder="50"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="batch_size">Batch Size</Label>
              <Input
                id="batch_size"
                type="number"
                min="1"
                max="100"
                value={formData.batch_size || '10'}
                onChange={(e) => handleInputChange('batch_size', e.target.value)}
                placeholder="10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="cleanup_interval">Cleanup Interval (hours)</Label>
              <Input
                id="cleanup_interval"
                type="number"
                min="1"
                max="168"
                value={formData.cleanup_interval || '24'}
                onChange={(e) => handleInputChange('cleanup_interval', e.target.value)}
                placeholder="24"
              />
            </div>
          </div>
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
