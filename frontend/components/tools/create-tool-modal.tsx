/**
 * Create MCP Tool Modal
 * =====================
 * Phase 3: Tools Integration
 * 
 * Form for creating new MCP tools with validation and JSON editors
 */

'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { X, Zap, Save, AlertCircle } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { useCreateMCPTool, useTestMCPToolConnection } from '@/hooks/use-mcp-tools-api'

interface CreateToolModalProps {
  open: boolean
  onClose: () => void
}

const TOOL_CATEGORIES = [
  { value: 'developer', label: 'Developer Tools', icon: '⚡' },
  { value: 'communication', label: 'Communication', icon: '💬' },
  { value: 'cloud', label: 'Cloud Services', icon: '☁️' },
  { value: 'analytics', label: 'Analytics', icon: '📊' },
  { value: 'productivity', label: 'Productivity', icon: '✨' },
  { value: 'security', label: 'Security', icon: '🛡️' },
  { value: 'data_analysis', label: 'Data Analysis', icon: '📈' },
  { value: 'automation', label: 'Automation', icon: '🤖' },
]

const COMMON_ICONS = ['⚡', '💬', '☁️', '📊', '✨', '🛡️', '📈', '🤖', '🔧', '🚀', '🔐', '📝', '🐳', '📁', '🔍', '⚙️']

export function CreateToolModal({ open, onClose }: CreateToolModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    mcp_server_url: '',
    provider: '',
    version: '1.0.0',
    icon: '⚡',
    category: 'developer',
    status: 'active',
    capabilities: '{\n  "methods": ["example.method"]\n}',
    credentials_schema: '{\n  "required": ["api_key"],\n  "optional": []\n}',
    tags: '',
    metadata: '{}',
  })

  const [errors, setErrors] = useState<Record<string, string>>({})
  const createMutation = useCreateMCPTool()
  const testMutation = useTestMCPToolConnection()

  const validateJSON = (field: string, value: string) => {
    try {
      JSON.parse(value)
      setErrors(prev => ({ ...prev, [field]: '' }))
      return true
    } catch (e: any) {
      setErrors(prev => ({ ...prev, [field]: `Invalid JSON: ${e.message}` }))
      return false
    }
  }

  const validateForm = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.name.trim()) {
      newErrors.name = 'Name is required'
    }
    if (!formData.provider.trim()) {
      newErrors.provider = 'Provider is required'
    }

    // Validate JSON fields
    if (!validateJSON('capabilities', formData.capabilities)) {
      newErrors.capabilities = errors.capabilities || 'Invalid JSON'
    }
    if (!validateJSON('credentials_schema', formData.credentials_schema)) {
      newErrors.credentials_schema = errors.credentials_schema || 'Invalid JSON'
    }
    if (!validateJSON('metadata', formData.metadata)) {
      newErrors.metadata = errors.metadata || 'Invalid JSON'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async () => {
    if (!validateForm()) return

    try {
      await createMutation.mutateAsync({
        name: formData.name,
        description: formData.description,
        mcp_server_url: formData.mcp_server_url || null,
        provider: formData.provider,
        version: formData.version,
        icon: formData.icon,
        category: formData.category,
        status: formData.status,
        capabilities: JSON.parse(formData.capabilities),
        credentials_schema: JSON.parse(formData.credentials_schema),
        tags: formData.tags ? formData.tags.split(',').map(t => t.trim()) : [],
        metadata: JSON.parse(formData.metadata),
      })

      handleClose()
    } catch (error: any) {
      console.error('Failed to create tool:', error)
    }
  }

  const handleTestConnection = async () => {
    if (!formData.mcp_server_url) {
      setErrors(prev => ({ ...prev, mcp_server_url: 'URL required for testing' }))
      return
    }

    // For testing, we need a tool ID. This would require creating the tool first.
    // For now, show a message
    alert('Connection test will be available after tool creation')
  }

  const handleClose = () => {
    setFormData({
      name: '',
      description: '',
      mcp_server_url: '',
      provider: '',
      version: '1.0.0',
      icon: '⚡',
      category: 'developer',
      status: 'active',
      capabilities: '{\n  "methods": ["example.method"]\n}',
      credentials_schema: '{\n  "required": ["api_key"],\n  "optional": []\n}',
      tags: '',
      metadata: '{}',
    })
    setErrors({})
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-brand-primary" />
            Create MCP Tool
          </DialogTitle>
          <DialogDescription>
            Add a new Model Context Protocol tool to the platform
          </DialogDescription>
        </DialogHeader>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Basic Information */}
          <div className="space-y-4">
            <h4 className="font-medium text-sm">Basic Information</h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name">
                  Tool Name <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., GitHub MCP"
                  className={errors.name ? 'border-destructive' : ''}
                />
                {errors.name && (
                  <p className="text-xs text-destructive">{errors.name}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="provider">
                  Provider <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="provider"
                  value={formData.provider}
                  onChange={(e) => setFormData({ ...formData, provider: e.target.value })}
                  placeholder="e.g., GitHub"
                  className={errors.provider ? 'border-destructive' : ''}
                />
                {errors.provider && (
                  <p className="text-xs text-destructive">{errors.provider}</p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Describe what this tool does"
                rows={2}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="category">Category</Label>
                <Select
                  value={formData.category}
                  onValueChange={(value) => setFormData({ ...formData, category: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TOOL_CATEGORIES.map(cat => (
                      <SelectItem key={cat.value} value={cat.value}>
                        <span className="flex items-center gap-2">
                          <span>{cat.icon}</span>
                          <span>{cat.label}</span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="version">Version</Label>
                <Input
                  id="version"
                  value={formData.version}
                  onChange={(e) => setFormData({ ...formData, version: e.target.value })}
                  placeholder="1.0.0"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="status">Status</Label>
                <Select
                  value={formData.status}
                  onValueChange={(value) => setFormData({ ...formData, status: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="inactive">Inactive</SelectItem>
                    <SelectItem value="deprecated">Deprecated</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="icon">Icon (Emoji)</Label>
              <div className="flex gap-2">
                <Input
                  id="icon"
                  value={formData.icon}
                  onChange={(e) => setFormData({ ...formData, icon: e.target.value })}
                  placeholder="⚡"
                  className="w-20"
                  maxLength={2}
                />
                <div className="flex gap-1 flex-wrap">
                  {COMMON_ICONS.map(icon => (
                    <Button
                      key={icon}
                      type="button"
                      variant="outline"
                      size="sm"
                      className="text-lg px-2"
                      onClick={() => setFormData({ ...formData, icon })}
                    >
                      {icon}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Connection Details */}
          <div className="space-y-4">
            <h4 className="font-medium text-sm">Connection Details</h4>
            
            <div className="space-y-2">
              <Label htmlFor="mcp_server_url">MCP Server URL</Label>
              <div className="flex gap-2">
                <Input
                  id="mcp_server_url"
                  value={formData.mcp_server_url}
                  onChange={(e) => setFormData({ ...formData, mcp_server_url: e.target.value })}
                  placeholder="http://localhost:3000/mcp/tool"
                  className={errors.mcp_server_url ? 'border-destructive' : ''}
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleTestConnection}
                  disabled={testMutation.isPending}
                >
                  {testMutation.isPending ? 'Testing...' : 'Test'}
                </Button>
              </div>
              {errors.mcp_server_url && (
                <p className="text-xs text-destructive">{errors.mcp_server_url}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="tags">Tags (comma-separated)</Label>
              <Input
                id="tags"
                value={formData.tags}
                onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
                placeholder="github, version_control, api"
              />
              {formData.tags && (
                <div className="flex gap-1 flex-wrap">
                  {formData.tags.split(',').map((tag, i) => (
                    <Badge key={i} variant="secondary" className="text-xs">
                      {tag.trim()}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Capabilities JSON */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-sm">Capabilities (JSON)</h4>
              {errors.capabilities && (
                <div className="flex items-center gap-1 text-destructive text-xs">
                  <AlertCircle className="w-3 h-3" />
                  <span>{errors.capabilities}</span>
                </div>
              )}
            </div>
            <Textarea
              value={formData.capabilities}
              onChange={(e) => {
                setFormData({ ...formData, capabilities: e.target.value })
                validateJSON('capabilities', e.target.value)
              }}
              placeholder='{"methods": ["repos.list", "repos.create"]}'
              rows={4}
              className={`font-mono text-xs ${errors.capabilities ? 'border-destructive' : ''}`}
            />
            <p className="text-xs text-muted-foreground">
              Define the methods and parameters available for this tool
            </p>
          </div>

          {/* Credentials Schema JSON */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-sm">Credentials Schema (JSON)</h4>
              {errors.credentials_schema && (
                <div className="flex items-center gap-1 text-destructive text-xs">
                  <AlertCircle className="w-3 h-3" />
                  <span>{errors.credentials_schema}</span>
                </div>
              )}
            </div>
            <Textarea
              value={formData.credentials_schema}
              onChange={(e) => {
                setFormData({ ...formData, credentials_schema: e.target.value })
                validateJSON('credentials_schema', e.target.value)
              }}
              placeholder='{"required": ["api_token"], "optional": ["webhook_url"]}'
              rows={4}
              className={`font-mono text-xs ${errors.credentials_schema ? 'border-destructive' : ''}`}
            />
            <p className="text-xs text-muted-foreground">
              Define the credentials required for this tool
            </p>
          </div>

          {/* Metadata JSON */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-sm">Metadata (JSON) - Optional</h4>
              {errors.metadata && (
                <div className="flex items-center gap-1 text-destructive text-xs">
                  <AlertCircle className="w-3 h-3" />
                  <span>{errors.metadata}</span>
                </div>
              )}
            </div>
            <Textarea
              value={formData.metadata}
              onChange={(e) => {
                setFormData({ ...formData, metadata: e.target.value })
                validateJSON('metadata', e.target.value)
              }}
              placeholder='{"documentation": "https://example.com/docs"}'
              rows={3}
              className={`font-mono text-xs ${errors.metadata ? 'border-destructive' : ''}`}
            />
          </div>
        </motion.div>

        <DialogFooter className="flex justify-between">
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={createMutation.isPending}
            className="bg-brand-primary hover:bg-brand-primary/90"
          >
            {createMutation.isPending ? (
              <>Creating...</>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Create Tool
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

