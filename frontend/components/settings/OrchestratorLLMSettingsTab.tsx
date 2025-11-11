/**
 * Orchestrator LLM Settings Tab Component
 * ======================================
 * 
 * Manages LLM configuration for the orchestrator system.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw, Brain, Zap, Settings } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface OrchestratorLLMSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function OrchestratorLLMSettingsTab({ 
  settings, 
  onSave, 
  saving, 
  onReset 
}: OrchestratorLLMSettingsTabProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})

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
            Configure the default LLM provider and model for orchestrator operations
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
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4">GPT-4</SelectItem>
                  <SelectItem value="gpt-4o">GPT-4o (128K context)</SelectItem>
                  <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                  <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                  <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                  <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                  <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
                  <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                  <SelectItem value="gemini-pro-vision">Gemini Pro Vision</SelectItem>
                </SelectContent>
              </Select>
              {getSetting('llm_model')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
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
