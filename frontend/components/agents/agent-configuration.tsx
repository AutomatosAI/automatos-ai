
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Settings, 
  Save, 
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Info,
  Cpu,
  Memory,
  Clock,
  Zap,
  Shield,
  Database
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'react-hot-toast'

// API hooks
import { useAgent, useAgentConfig, useUpdateAgentConfig } from '@/hooks/use-agent-api'

interface AgentConfigurationProps {
  agents: any[]
  selectedAgentId: string | null
  onAgentSelect: (agentId: string | null) => void
}

export function AgentConfiguration({ 
  agents, 
  selectedAgentId, 
  onAgentSelect 
}: AgentConfigurationProps) {
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [configData, setConfigData] = useState<any>({})

  // Fetch agent and configuration data
  const { data: agent, isLoading: agentLoading } = useAgent(selectedAgentId)
  const { data: agentConfig, isLoading: configLoading } = useAgentConfig(selectedAgentId)
  const updateConfigMutation = useUpdateAgentConfig()

  // Initialize config data when agent config is loaded
  useEffect(() => {
    if (agentConfig) {
      setConfigData(agentConfig)
      setHasUnsavedChanges(false)
    }
  }, [agentConfig])

  // Handle form changes
  const handleConfigChange = (key: string, value: any) => {
    setConfigData((prev: any) => ({
      ...prev,
      [key]: value
    }))
    setHasUnsavedChanges(true)
  }

  // Handle nested config changes
  const handleNestedConfigChange = (section: string, key: string, value: any) => {
    setConfigData((prev: any) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }))
    setHasUnsavedChanges(true)
  }

  // Save configuration
  const handleSave = async () => {
    if (!selectedAgentId) return

    try {
      await updateConfigMutation.mutateAsync({
        agentId: selectedAgentId,
        config: configData
      })
      setHasUnsavedChanges(false)
    } catch (error) {
      // Error already handled by the hook
    }
  }

  // Reset configuration
  const handleReset = () => {
    if (agentConfig) {
      setConfigData(agentConfig)
      setHasUnsavedChanges(false)
    }
  }

  if (!selectedAgentId) {
    return (
      <Card className="glass-card">
        <CardContent className="p-12 text-center">
          <Settings className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Select an Agent</h3>
          <p className="text-muted-foreground">
            Choose an agent to configure its settings and parameters
          </p>
        </CardContent>
      </Card>
    )
  }

  if (agentLoading || configLoading) {
    return (
      <div className="space-y-6">
        <Card className="glass-card">
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-64" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-20 w-full" />
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Configuration</h2>
          <p className="text-muted-foreground">
            Configure settings and parameters for {agent?.name}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {hasUnsavedChanges && (
            <Badge variant="secondary" className="text-yellow-600">
              <AlertTriangle className="w-3 h-3 mr-1" />
              Unsaved Changes
            </Badge>
          )}
          
          <Button 
            variant="outline" 
            onClick={handleReset}
            disabled={!hasUnsavedChanges || updateConfigMutation.isPending}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          
          <Button 
            onClick={handleSave}
            disabled={!hasUnsavedChanges || updateConfigMutation.isPending}
          >
            <Save className="w-4 h-4 mr-2" />
            {updateConfigMutation.isPending ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {/* Agent Info */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-white">
              🤖
            </div>
            <div>
              <div>{agent?.name}</div>
              <p className="text-sm font-normal text-muted-foreground capitalize">
                {agent?.agent_type?.replace('_', ' ')} • {agent?.status}
              </p>
            </div>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Configuration Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Basic Configuration */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Basic Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="agent-name">Agent Name</Label>
              <Input
                id="agent-name"
                value={configData.name || agent?.name || ''}
                onChange={(e) => handleConfigChange('name', e.target.value)}
                placeholder="Enter agent name"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="agent-description">Description</Label>
              <Textarea
                id="agent-description"
                value={configData.description || agent?.description || ''}
                onChange={(e) => handleConfigChange('description', e.target.value)}
                placeholder="Describe the agent's purpose and capabilities"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="priority-level">Priority Level</Label>
              <Select
                value={configData.priority_level || agent?.priority_level || 'medium'}
                onValueChange={(value) => handleConfigChange('priority_level', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low Priority</SelectItem>
                  <SelectItem value="medium">Medium Priority</SelectItem>
                  <SelectItem value="high">High Priority</SelectItem>
                  <SelectItem value="critical">Critical Priority</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto Start</Label>
                <p className="text-sm text-muted-foreground">
                  Start agent automatically on system boot
                </p>
              </div>
              <Switch
                checked={configData.auto_start || agent?.auto_start || false}
                onCheckedChange={(checked) => handleConfigChange('auto_start', checked)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Performance Settings */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Performance Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Max Concurrent Tasks</Label>
              <div className="px-3">
                <Slider
                  value={[configData.max_concurrent_tasks || agent?.max_concurrent_tasks || 5]}
                  onValueChange={(value) => handleConfigChange('max_concurrent_tasks', value[0])}
                  max={20}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-muted-foreground mt-1">
                  <span>1</span>
                  <span>Current: {configData.max_concurrent_tasks || agent?.max_concurrent_tasks || 5}</span>
                  <span>20</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Task Timeout (seconds)</Label>
              <Input
                type="number"
                value={configData.task_timeout || '300'}
                onChange={(e) => handleConfigChange('task_timeout', parseInt(e.target.value) || 300)}
                min="10"
                max="3600"
              />
            </div>

            <div className="space-y-2">
              <Label>Retry Attempts</Label>
              <Input
                type="number"
                value={configData.retry_attempts || '3'}
                onChange={(e) => handleConfigChange('retry_attempts', parseInt(e.target.value) || 3)}
                min="0"
                max="10"
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable Caching</Label>
                <p className="text-sm text-muted-foreground">
                  Cache responses to improve performance
                </p>
              </div>
              <Switch
                checked={configData.enable_caching || false}
                onCheckedChange={(checked) => handleConfigChange('enable_caching', checked)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Security & Access
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Access Level</Label>
              <Select
                value={configData.access_level || 'standard'}
                onValueChange={(value) => handleConfigChange('access_level', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="restricted">Restricted</SelectItem>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="elevated">Elevated</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable Logging</Label>
                <p className="text-sm text-muted-foreground">
                  Log all agent activities
                </p>
              </div>
              <Switch
                checked={configData.enable_logging !== false}
                onCheckedChange={(checked) => handleConfigChange('enable_logging', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Rate Limiting</Label>
                <p className="text-sm text-muted-foreground">
                  Apply rate limits to API calls
                </p>
              </div>
              <Switch
                checked={configData.enable_rate_limiting || true}
                onCheckedChange={(checked) => handleConfigChange('enable_rate_limiting', checked)}
              />
            </div>

            <div className="space-y-2">
              <Label>API Rate Limit (calls/minute)</Label>
              <Input
                type="number"
                value={configData.api_rate_limit || '100'}
                onChange={(e) => handleConfigChange('api_rate_limit', parseInt(e.target.value) || 100)}
                min="1"
                max="1000"
              />
            </div>
          </CardContent>
        </Card>

        {/* Memory & Storage */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Memory & Storage
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Memory Limit (MB)</Label>
              <div className="px-3">
                <Slider
                  value={[configData.memory_limit || 512]}
                  onValueChange={(value) => handleConfigChange('memory_limit', value[0])}
                  max={4096}
                  min={128}
                  step={128}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-muted-foreground mt-1">
                  <span>128MB</span>
                  <span>Current: {configData.memory_limit || 512}MB</span>
                  <span>4GB</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Context Window Size</Label>
              <Select
                value={configData.context_window_size?.toString() || '8192'}
                onValueChange={(value) => handleConfigChange('context_window_size', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2048">2K Tokens</SelectItem>
                  <SelectItem value="4096">4K Tokens</SelectItem>
                  <SelectItem value="8192">8K Tokens</SelectItem>
                  <SelectItem value="16384">16K Tokens</SelectItem>
                  <SelectItem value="32768">32K Tokens</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Persistent Memory</Label>
                <p className="text-sm text-muted-foreground">
                  Retain memory between sessions
                </p>
              </div>
              <Switch
                checked={configData.persistent_memory || true}
                onCheckedChange={(checked) => handleConfigChange('persistent_memory', checked)}
              />
            </div>

            <div className="space-y-2">
              <Label>Memory Cleanup Interval (hours)</Label>
              <Input
                type="number"
                value={configData.memory_cleanup_interval || '24'}
                onChange={(e) => handleConfigChange('memory_cleanup_interval', parseInt(e.target.value) || 24)}
                min="1"
                max="168"
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Advanced Configuration */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Advanced Configuration</CardTitle>
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              Advanced settings should only be modified by experienced users. Changes may affect agent performance.
            </AlertDescription>
          </Alert>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="custom-config">Custom Configuration (JSON)</Label>
            <Textarea
              id="custom-config"
              value={JSON.stringify(configData.custom_config || {}, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value)
                  handleConfigChange('custom_config', parsed)
                } catch (error) {
                  // Invalid JSON, don't update
                }
              }}
              placeholder='{"key": "value"}'
              rows={6}
              className="font-mono text-sm"
            />
          </div>
        </CardContent>
      </Card>

      {/* Save Status */}
      {hasUnsavedChanges && (
        <Alert className="border-yellow-200 bg-yellow-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            You have unsaved changes. Don't forget to save your configuration.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}

