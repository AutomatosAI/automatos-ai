/**
 * General Settings Tab Component
 * =============================
 * 
 * Manages general system settings like environment, logging, deployment, etc.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw, Globe, Database, Zap, Shield } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface GeneralSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function GeneralSettingsTab({ 
  settings, 
  onSave, 
  saving, 
  onReset 
}: GeneralSettingsTabProps) {
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
      {/* Environment Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5" />
            Environment Configuration
          </CardTitle>
          <CardDescription>
            Core environment settings for the application
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="environment">Environment</Label>
              <Select 
                value={formData.environment || ''} 
                onValueChange={(value) => handleInputChange('environment', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select environment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="development">Development</SelectItem>
                  <SelectItem value="staging">Staging</SelectItem>
                  <SelectItem value="production">Production</SelectItem>
                </SelectContent>
              </Select>
              {getSetting('environment')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_level">Log Level</Label>
              <Select 
                value={formData.log_level || ''} 
                onValueChange={(value) => handleInputChange('log_level', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select log level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="DEBUG">DEBUG</SelectItem>
                  <SelectItem value="INFO">INFO</SelectItem>
                  <SelectItem value="WARNING">WARNING</SelectItem>
                  <SelectItem value="ERROR">ERROR</SelectItem>
                  <SelectItem value="CRITICAL">CRITICAL</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ML/AI Model Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            ML/AI Model Configuration
          </CardTitle>
          <CardDescription>
            Embedding models and vector store settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="embedding_model">Embedding Model</Label>
              <Input
                id="embedding_model"
                value={formData.embedding_model || ''}
                onChange={(e) => handleInputChange('embedding_model', e.target.value)}
                placeholder="e.g., disabled, openai, local"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="embedding_cache_dir">Cache Directory</Label>
              <Input
                id="embedding_cache_dir"
                value={formData.embedding_cache_dir || ''}
                onChange={(e) => handleInputChange('embedding_cache_dir', e.target.value)}
                placeholder="./model_cache"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="embedding_max_seq_length">Max Sequence Length</Label>
              <Input
                id="embedding_max_seq_length"
                type="number"
                value={formData.embedding_max_seq_length || ''}
                onChange={(e) => handleInputChange('embedding_max_seq_length', e.target.value)}
                placeholder="256"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="openai_embedding_model">OpenAI Embedding Model</Label>
              <Input
                id="openai_embedding_model"
                value={formData.openai_embedding_model || ''}
                onChange={(e) => handleInputChange('openai_embedding_model', e.target.value)}
                placeholder="text-embedding-3-small"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="vector_store_type">Vector Store Type</Label>
              <Select 
                value={formData.vector_store_type || ''} 
                onValueChange={(value) => handleInputChange('vector_store_type', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select vector store type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="faiss">FAISS</SelectItem>
                  <SelectItem value="chroma">Chroma</SelectItem>
                  <SelectItem value="pinecone">Pinecone</SelectItem>
                  <SelectItem value="weaviate">Weaviate</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="vector_store_dimensions">Vector Dimensions</Label>
              <Input
                id="vector_store_dimensions"
                type="number"
                value={formData.vector_store_dimensions || ''}
                onChange={(e) => handleInputChange('vector_store_dimensions', e.target.value)}
                placeholder="384"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="chunk_size">Chunk Size</Label>
              <Input
                id="chunk_size"
                type="number"
                value={formData.chunk_size || ''}
                onChange={(e) => handleInputChange('chunk_size', e.target.value)}
                placeholder="512"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="chunk_overlap">Chunk Overlap</Label>
              <Input
                id="chunk_overlap"
                type="number"
                value={formData.chunk_overlap || ''}
                onChange={(e) => handleInputChange('chunk_overlap', e.target.value)}
                placeholder="50"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max_context_length">Max Context Length</Label>
              <Input
                id="max_context_length"
                type="number"
                value={formData.max_context_length || ''}
                onChange={(e) => handleInputChange('max_context_length', e.target.value)}
                placeholder="4000"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Deployment Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Deployment Configuration
          </CardTitle>
          <CardDescription>
            SSH deployment and frontend settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="deploy_host">Deploy Host</Label>
              <Input
                id="deploy_host"
                value={formData.deploy_host || ''}
                onChange={(e) => handleInputChange('deploy_host', e.target.value)}
                placeholder="mcp.automatos.app"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="deploy_port">Deploy Port</Label>
              <Input
                id="deploy_port"
                type="number"
                value={formData.deploy_port || ''}
                onChange={(e) => handleInputChange('deploy_port', e.target.value)}
                placeholder="22"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="deploy_user">Deploy User</Label>
              <Input
                id="deploy_user"
                value={formData.deploy_user || ''}
                onChange={(e) => handleInputChange('deploy_user', e.target.value)}
                placeholder="root"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="deploy_key_path">Deploy Key Path</Label>
              <Input
                id="deploy_key_path"
                value={formData.deploy_key_path || ''}
                onChange={(e) => handleInputChange('deploy_key_path', e.target.value)}
                placeholder="/aRpp/keys/deploy_key"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="deploy_enabled">Deploy Enabled</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={formData.deploy_enabled === 'true'}
                  onCheckedChange={(checked) => handleInputChange('deploy_enabled', checked.toString())}
                />
                <Label>Enable deployment</Label>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Frontend Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Frontend Configuration
          </CardTitle>
          <CardDescription>
            NextJS frontend and authentication settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="nextauth_secret">NextAuth Secret</Label>
              <Input
                id="nextauth_secret"
                type="password"
                value={formData.nextauth_secret || ''}
                onChange={(e) => handleInputChange('nextauth_secret', e.target.value)}
                placeholder="your-nextauth-secret-here"
              />
              {getSetting('nextauth_secret')?.is_sensitive && (
                <Badge variant="destructive" className="text-xs">Sensitive</Badge>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="nextauth_url">NextAuth URL</Label>
              <Input
                id="nextauth_url"
                value={formData.nextauth_url || ''}
                onChange={(e) => handleInputChange('nextauth_url', e.target.value)}
                placeholder="https://8018adcb5.preview.abacusai.app"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="next_public_api_url">Public API URL</Label>
              <Input
                id="next_public_api_url"
                value={formData.next_public_api_url || ''}
                onChange={(e) => handleInputChange('next_public_api_url', e.target.value)}
                placeholder="https://api.automatos.app"
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
