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
import { Save, RotateCcw, Globe, Database, Zap, Shield, Key } from 'lucide-react'
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

  const MODEL_DIMENSIONS: Record<string, Record<string, number>> = {
    openrouter: {
      'qwen/qwen3-embedding-8b': 4096,
      'qwen/qwen3-embedding-4b': 2560,
      'google/gemini-embedding-001': 3072,
      'openai/text-embedding-3-large': 3072,
      'openai/text-embedding-3-small': 1536,
      'openai/text-embedding-ada-002': 1536,
      'mistralai/mistral-embed-2312': 1024,
      'mistralai/codestral-embed-2505': 1024,
      'baai/bge-m3': 1024,
      'baai/bge-large-en-v1.5': 1024,
      'intfloat/e5-large-v2': 1024,
      'intfloat/multilingual-e5-large': 1024,
      'sentence-transformers/all-mpnet-base-v2': 768,
      'sentence-transformers/all-minilm-l6-v2': 384,
      'thenlper/gte-large': 1024,
    },
    openai: {
      'text-embedding-3-small': 1536,
      'text-embedding-3-large': 3072,
      'text-embedding-ada-002': 1536
    },
    google: {
      'text-embedding-004': 768,
      'textembedding-gecko': 768
    },
    cohere: {
      'embed-english-v3.0': 1024,
      'embed-multilingual-v3.0': 1024
    },
    huggingface_local: {
      'all-MiniLM-L6-v2': 384,
      'all-mpnet-base-v2': 768,
      'BAAI/bge-large-en-v1.5': 1024,
      'BAAI/bge-m3': 1024,
      'nomic-ai/nomic-embed-text-v1.5': 768,
      'intfloat/e5-large-v2': 1024,
      'Alibaba-NLP/gte-large-en-v1.5': 1024
    },
    huggingface_api: {
      'sentence-transformers/all-MiniLM-L6-v2': 384,
      'sentence-transformers/all-mpnet-base-v2': 768,
      'BAAI/bge-large-en-v1.5': 1024,
      'BAAI/bge-m3': 1024,
      'nomic-ai/nomic-embed-text-v1.5': 768,
      'intfloat/e5-large-v2': 1024
    }
  }

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

  React.useEffect(() => {
    const provider = formData.embedding_provider
    const model = formData.embedding_model
    if (!provider || !model) return
    const mappedDimension = MODEL_DIMENSIONS[provider]?.[model]
    if (!mappedDimension) return
    if (formData.vector_store_dimensions !== mappedDimension.toString()) {
      setFormData(prev => ({ ...prev, vector_store_dimensions: mappedDimension.toString() }))
    }
  }, [formData.embedding_provider, formData.embedding_model])

  const handleInputChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const applyModelSelection = (provider: string, model: string) => {
    handleInputChange('embedding_model', model)
    if (provider === 'openai') handleInputChange('openai_embedding_model', model)
    const mappedDimension = MODEL_DIMENSIONS[provider]?.[model]
    if (mappedDimension) handleInputChange('vector_store_dimensions', mappedDimension.toString())
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
            Global embedding configuration (used across all features: RAG, memory, search)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Embedding Provider Selector */}
            <div className="space-y-2">
              <Label htmlFor="embedding_provider">Embedding Provider</Label>
              <Select
                value={formData.embedding_provider || ''}
                onValueChange={(value) => {
                  handleInputChange('embedding_provider', value)
                  const defaultModel = Object.keys(MODEL_DIMENSIONS[value] || {})[0]
                  if (defaultModel) applyModelSelection(value, defaultModel)
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select embedding provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openrouter">OpenRouter (20+ Models, Parallel Batch)</SelectItem>
                  <SelectItem value="openai">OpenAI (Direct)</SelectItem>
                  <SelectItem value="google">Google</SelectItem>
                  <SelectItem value="cohere">Cohere</SelectItem>
                  <SelectItem value="huggingface_local">HuggingFace Local (FREE, Slow CPU)</SelectItem>
                  <SelectItem value="huggingface_api">HuggingFace API</SelectItem>
                  <SelectItem value="disabled">Disabled (Deterministic Fallback)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {formData.embedding_provider === 'openrouter'
                  ? 'OpenRouter: 20+ models via single API key. Parallel batch processing. Uses your OPENROUTER_API_KEY.'
                  : 'Choose your embedding provider. OpenRouter recommended for speed + quality.'}
              </p>
            </div>

            {/* Dynamic Embedding Model Selector */}
            <div className="space-y-2">
              <Label htmlFor="embedding_model">Embedding Model</Label>
              <Select
                value={formData.embedding_model || ''}
                onValueChange={(value) => {
                  const provider = formData.embedding_provider
                  if (provider) applyModelSelection(provider, value)
                  else handleInputChange('embedding_model', value)
                }}
                disabled={!formData.embedding_provider || formData.embedding_provider === 'disabled'}
              >
                <SelectTrigger>
                  <SelectValue placeholder={
                    !formData.embedding_provider
                      ? "Select provider first"
                      : "Select embedding model"
                  } />
                </SelectTrigger>
                <SelectContent>
                  {formData.embedding_provider === 'openrouter' && (
                    <>
                      <SelectItem value="qwen/qwen3-embedding-8b">Qwen3 8B (4096d, 32K ctx, $0.01/M) — Best Value</SelectItem>
                      <SelectItem value="qwen/qwen3-embedding-4b">Qwen3 4B (2560d, 32K ctx, $0.02/M) — Lighter</SelectItem>
                      <SelectItem value="google/gemini-embedding-001">Gemini Embedding (3072d, 20K ctx, $0.15/M) — #1 MTEB</SelectItem>
                      <SelectItem value="openai/text-embedding-3-large">OpenAI Large (3072d, 8K ctx, $0.13/M)</SelectItem>
                      <SelectItem value="openai/text-embedding-3-small">OpenAI Small (1536d, 8K ctx, $0.02/M)</SelectItem>
                      <SelectItem value="mistralai/mistral-embed-2312">Mistral Embed (1024d, 8K ctx, $0.10/M)</SelectItem>
                      <SelectItem value="mistralai/codestral-embed-2505">Codestral Embed (1024d, 8K ctx, $0.15/M) — Code</SelectItem>
                      <SelectItem value="baai/bge-m3">BGE-M3 (1024d, 8K ctx, $0.01/M) — Multilingual</SelectItem>
                      <SelectItem value="baai/bge-large-en-v1.5">BGE-Large (1024d, 512 ctx, $0.01/M)</SelectItem>
                      <SelectItem value="intfloat/e5-large-v2">E5-Large (1024d, 512 ctx, $0.01/M)</SelectItem>
                      <SelectItem value="intfloat/multilingual-e5-large">E5-Large Multilingual (1024d, 512 ctx, $0.01/M)</SelectItem>
                      <SelectItem value="sentence-transformers/all-mpnet-base-v2">MPNet-Base (768d, 512 ctx, $0.005/M)</SelectItem>
                      <SelectItem value="sentence-transformers/all-minilm-l6-v2">MiniLM-L6 (384d, 512 ctx, $0.005/M) — Fastest</SelectItem>
                      <SelectItem value="thenlper/gte-large">GTE-Large (1024d, 512 ctx, $0.01/M)</SelectItem>
                      <SelectItem value="openai/text-embedding-ada-002">Ada 002 (1536d, 8K ctx, $0.10/M) — Legacy</SelectItem>
                    </>
                  )}
                  {formData.embedding_provider === 'openai' && (
                    <>
                      <SelectItem value="text-embedding-3-small">text-embedding-3-small (384 dims, fast)</SelectItem>
                      <SelectItem value="text-embedding-3-large">text-embedding-3-large (1536 dims, quality)</SelectItem>
                      <SelectItem value="text-embedding-ada-002">text-embedding-ada-002 (1536 dims, legacy)</SelectItem>
                    </>
                  )}
                  {formData.embedding_provider === 'google' && (
                    <>
                      <SelectItem value="text-embedding-004">text-embedding-004 (768 dims)</SelectItem>
                      <SelectItem value="textembedding-gecko">textembedding-gecko (768 dims, legacy)</SelectItem>
                    </>
                  )}
                  {formData.embedding_provider === 'cohere' && (
                    <>
                      <SelectItem value="embed-english-v3.0">embed-english-v3.0 (1024 dims)</SelectItem>
                      <SelectItem value="embed-multilingual-v3.0">embed-multilingual-v3.0 (1024 dims)</SelectItem>
                    </>
                  )}
                  {formData.embedding_provider === 'huggingface_local' && (
                    <>
                      <SelectItem value="BAAI/bge-large-en-v1.5">⭐ BGE-Large (1024d) - Best Quality</SelectItem>
                      <SelectItem value="BAAI/bge-m3">⭐ BGE-M3 (1024d) - Best + Multilingual</SelectItem>
                      <SelectItem value="nomic-ai/nomic-embed-text-v1.5">Nomic (768d) - 8K Context Length</SelectItem>
                      <SelectItem value="intfloat/e5-large-v2">E5-Large (1024d) - Great Quality</SelectItem>
                      <SelectItem value="Alibaba-NLP/gte-large-en-v1.5">GTE-Large (1024d) - Top MTEB</SelectItem>
                      <SelectItem value="all-mpnet-base-v2">MPNet-Base (768d) - Balanced</SelectItem>
                      <SelectItem value="all-MiniLM-L6-v2">MiniLM (384d) - Fast/Light</SelectItem>
                    </>
                  )}
                  {formData.embedding_provider === 'huggingface_api' && (
                    <>
                      <SelectItem value="BAAI/bge-large-en-v1.5">⭐ BGE-Large (1024d) - Best Quality</SelectItem>
                      <SelectItem value="BAAI/bge-m3">⭐ BGE-M3 (1024d) - Best + Multilingual</SelectItem>
                      <SelectItem value="nomic-ai/nomic-embed-text-v1.5">Nomic (768d) - 8K Context</SelectItem>
                      <SelectItem value="intfloat/e5-large-v2">E5-Large (1024d) - Great Quality</SelectItem>
                      <SelectItem value="sentence-transformers/all-mpnet-base-v2">MPNet-Base (768d)</SelectItem>
                      <SelectItem value="sentence-transformers/all-MiniLM-L6-v2">MiniLM (384d) - Fast</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>
              {formData.embedding_provider === 'disabled' && (
                <p className="text-xs text-amber-600">
                  ⚠️ Using deterministic fallback (hash-based, no semantic meaning). Configure a provider for real embeddings.
                </p>
              )}
            </div>

            {/* Cache Directory (for local models) */}
            <div className="space-y-2">
              <Label htmlFor="embedding_cache_dir">Cache Directory</Label>
              <Input
                id="embedding_cache_dir"
                value={formData.embedding_cache_dir || ''}
                onChange={(e) => handleInputChange('embedding_cache_dir', e.target.value)}
                placeholder="./model_cache"
                disabled={formData.embedding_provider !== 'huggingface_local'}
              />
              <p className="text-xs text-muted-foreground">
                {formData.embedding_provider === 'huggingface_local'
                  ? 'Local directory to cache downloaded models'
                  : 'Only used for local models'
                }
              </p>
            </div>

            {/* Vector Dimensions (auto-populated based on model) */}
            <div className="space-y-2">
              <Label htmlFor="vector_store_dimensions">Vector Dimensions (auto)</Label>
              <Input
                id="vector_store_dimensions"
                type="number"
                value={formData.vector_store_dimensions || ''}
                onChange={(e) => handleInputChange('vector_store_dimensions', e.target.value)}
                placeholder="384"
                className="bg-muted"
              />
              <p className="text-xs text-muted-foreground">
                Automatically set based on your embedding model choice
              </p>
            </div>

            {/* Vector Store Type */}
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
                  <SelectItem value="pgvector">PgVector (Recommended)</SelectItem>
                  <SelectItem value="faiss">FAISS</SelectItem>
                  <SelectItem value="chroma">Chroma</SelectItem>
                  <SelectItem value="pinecone">Pinecone</SelectItem>
                  <SelectItem value="weaviate">Weaviate</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Max Sequence Length */}
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

            {/* Chunk Size */}
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

            {/* Chunk Overlap */}
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

            {/* Max Context Length */}
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
                placeholder="your-deploy-host.com"
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
                placeholder="/path/to/your/deploy_key"
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
            NextJS frontend settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="next_public_api_url">Public API URL</Label>
              <Input
                id="next_public_api_url"
                value={formData.next_public_api_url || ''}
                onChange={(e) => handleInputChange('next_public_api_url', e.target.value)}
                placeholder="https://your-api-url.com"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Legacy tool gateway settings removed (Composio-backed tools cache is source-of-truth) */}

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
