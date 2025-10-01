

'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Settings, 
  Key, 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  Eye, 
  EyeOff,
  Globe,
  Server,
  Lock,
  Zap,
  AlertCircle,
  Save,
  TestTube,
  RefreshCw,
  Copy,
  ExternalLink
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

interface Tool {
  id: number
  name: string
  description: string
  category: string
  icon: string
  provider: string
  status: string
  isInstalled: boolean
  isConfigured: boolean
  version: string
  pricing: string
  rating: number
  usageCount: number
  tags: string[]
  permissions: string[]
  requiredCredentials: string[]
  supportedEnvironments: string[]
  lastUpdated: string
  configuration?: Record<string, any>
}

interface ToolConfigModalProps {
  open: boolean
  onClose: () => void
  tool: Tool | null
  onSave: (toolId: number, config: Record<string, any>) => void
}

const environmentColors = {
  development: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  staging: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  production: 'bg-green-500/10 text-green-400 border-green-500/20'
}

const credentialTypes = {
  api_token: { label: 'API Token', type: 'password', icon: Key },
  api_key: { label: 'API Key', type: 'password', icon: Key },
  secret_key: { label: 'Secret Key', type: 'password', icon: Lock },
  oauth2_token: { label: 'OAuth2 Token', type: 'password', icon: Shield },
  webhook_url: { label: 'Webhook URL', type: 'url', icon: Globe },
  bot_token: { label: 'Bot Token', type: 'password', icon: Key },
  domain: { label: 'Domain', type: 'text', icon: Server },
  region: { label: 'Region', type: 'text', icon: Globe },
  access_key: { label: 'Access Key', type: 'password', icon: Key },
  property_id: { label: 'Property ID', type: 'text', icon: Server },
  service_account_key: { label: 'Service Account Key', type: 'textarea', icon: Key },
  app_key: { label: 'App Key', type: 'password', icon: Key },
  registry_token: { label: 'Registry Token', type: 'password', icon: Key }
}

export function ToolConfigModal({ open, onClose, tool, onSave }: ToolConfigModalProps) {
  const [activeTab, setActiveTab] = useState('credentials')
  const [configuration, setConfiguration] = useState<Record<string, any>>({})
  const [credentials, setCredentials] = useState<Record<string, any>>({})
  const [selectedEnvironment, setSelectedEnvironment] = useState('development')
  const [showCredentials, setShowCredentials] = useState<Record<string, boolean>>({})
  const [testConnection, setTestConnection] = useState<{
    testing: boolean
    status: 'success' | 'error' | null
    message: string
  }>({
    testing: false,
    status: null,
    message: ''
  })
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (tool) {
      // Initialize configuration from tool data
      setConfiguration(tool?.configuration || {})
      setCredentials({})
      setSelectedEnvironment(tool?.supportedEnvironments?.[0] || 'development')
      setShowCredentials({})
      setTestConnection({ testing: false, status: null, message: '' })
    }
  }, [tool])

  const handleCredentialChange = (key: string, value: string) => {
    setCredentials(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const handleConfigurationChange = (key: string, value: any) => {
    setConfiguration(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const toggleShowCredential = (key: string) => {
    setShowCredentials(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  const handleTestConnection = async () => {
    if (!tool) return

    setTestConnection(prev => ({ ...prev, testing: true }))

    try {
      // Simulate API call to test connection
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Mock success/failure based on credentials
      const hasRequiredCredentials = tool?.requiredCredentials?.every(cred => credentials[cred])
      
      if (hasRequiredCredentials) {
        setTestConnection({
          testing: false,
          status: 'success',
          message: 'Connection successful! All credentials are valid.'
        })
      } else {
        setTestConnection({
          testing: false,
          status: 'error',
          message: 'Connection failed. Please check your credentials.'
        })
      }
    } catch (error) {
      setTestConnection({
        testing: false,
        status: 'error',
        message: 'Connection test failed. Please try again.'
      })
    }
  }

  const handleSave = async () => {
    if (!tool) return

    setSaving(true)
    try {
      // Simulate API call to save configuration
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const finalConfig = {
        ...configuration,
        credentials: credentials,
        environment: selectedEnvironment,
        configured_at: new Date().toISOString()
      }

      onSave(tool.id, finalConfig)
      onClose()
    } catch (error) {
      console.error('Failed to save configuration:', error)
    } finally {
      setSaving(false)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  if (!tool) return null

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="glass-card max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
              <span className="text-lg">{tool?.icon}</span>
            </div>
            <div>
              <h2 className="text-xl font-semibold">Configure {tool?.name}</h2>
              <p className="text-sm text-muted-foreground font-normal">
                Set up credentials and configuration for {tool?.provider}
              </p>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-4 bg-secondary/50">
              <TabsTrigger value="credentials" className="flex items-center space-x-2">
                <Key className="w-4 h-4" />
                <span>Credentials</span>
              </TabsTrigger>
              <TabsTrigger value="configuration" className="flex items-center space-x-2">
                <Settings className="w-4 h-4" />
                <span>Configuration</span>
              </TabsTrigger>
              <TabsTrigger value="security" className="flex items-center space-x-2">
                <Shield className="w-4 h-4" />
                <span>Security</span>
              </TabsTrigger>
              <TabsTrigger value="testing" className="flex items-center space-x-2">
                <TestTube className="w-4 h-4" />
                <span>Testing</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="credentials" className="space-y-6">
              {/* Environment Selection */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Globe className="w-5 h-5" />
                    <span>Environment</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {tool?.supportedEnvironments?.map((env) => (
                      <Button
                        key={env}
                        variant={selectedEnvironment === env ? 'default' : 'outline'}
                        onClick={() => setSelectedEnvironment(env)}
                        className={`flex items-center space-x-2 ${
                          selectedEnvironment === env ? 'icon-gradient' : 'hover:border-orange-500/50'
                        }`}
                      >
                        <span>{env}</span>
                        {selectedEnvironment === env && <CheckCircle className="w-4 h-4" />}
                      </Button>
                    )) || []}
                  </div>
                </CardContent>
              </Card>

              {/* Credentials Form */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Key className="w-5 h-5" />
                    <span>API Credentials</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {tool?.requiredCredentials?.map((credKey) => {
                    const credType = credentialTypes[credKey as keyof typeof credentialTypes]
                    if (!credType) return null

                    const CredIcon = credType.icon
                    const isPassword = credType.type === 'password'
                    const showValue = showCredentials[credKey]

                    return (
                      <div key={credKey} className="space-y-2">
                        <Label htmlFor={credKey} className="flex items-center space-x-2">
                          <CredIcon className="w-4 h-4" />
                          <span>{credType.label}</span>
                          {tool?.requiredCredentials?.includes(credKey) && (
                            <Badge variant="outline" className="text-xs">Required</Badge>
                          )}
                        </Label>
                        <div className="relative">
                          {credType.type === 'textarea' ? (
                            <Textarea
                              id={credKey}
                              placeholder={`Enter your ${credType.label.toLowerCase()}`}
                              value={credentials[credKey] || ''}
                              onChange={(e) => handleCredentialChange(credKey, e.target.value)}
                              rows={3}
                            />
                          ) : (
                            <Input
                              id={credKey}
                              type={isPassword && !showValue ? 'password' : 'text'}
                              placeholder={`Enter your ${credType.label.toLowerCase()}`}
                              value={credentials[credKey] || ''}
                              onChange={(e) => handleCredentialChange(credKey, e.target.value)}
                            />
                          )}
                          {isPassword && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="absolute right-2 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0"
                              onClick={() => toggleShowCredential(credKey)}
                            >
                              {showValue ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </Button>
                          )}
                        </div>
                        {credentials[credKey] && (
                          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                            <CheckCircle className="w-3 h-3 text-green-400" />
                            <span>Credential provided</span>
                          </div>
                        )}
                      </div>
                    )
                  }) || []}

                  {/* Credential Help */}
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-sm">
                      All credentials are encrypted and stored securely. They are only accessible by agents with proper permissions.
                      <Button variant="link" className="p-0 h-auto text-xs ml-2" asChild>
                        <a href="#" target="_blank" rel="noopener noreferrer">
                          Learn more about {tool?.provider} API keys
                          <ExternalLink className="w-3 h-3 ml-1" />
                        </a>
                      </Button>
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="configuration" className="space-y-6">
              {/* Tool-specific Configuration */}
              {tool?.name === 'GitHub' && (
                <Card className="bg-secondary/30 border-border/30">
                  <CardHeader>
                    <CardTitle className="text-base">GitHub Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Webhook URL</Label>
                      <Input
                        value={configuration.webhook_url || ''}
                        onChange={(e) => handleConfigurationChange('webhook_url', e.target.value)}
                        placeholder="https://your-app.com/webhooks/github"
                      />
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={configuration.branch_protection || false}
                        onCheckedChange={(checked) => handleConfigurationChange('branch_protection', checked)}
                      />
                      <Label>Enable branch protection</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={configuration.auto_merge || false}
                        onCheckedChange={(checked) => handleConfigurationChange('auto_merge', checked)}
                      />
                      <Label>Enable auto-merge for approved PRs</Label>
                    </div>
                  </CardContent>
                </Card>
              )}

              {tool?.name === 'Slack' && (
                <Card className="bg-secondary/30 border-border/30">
                  <CardHeader>
                    <CardTitle className="text-base">Slack Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Default Channel</Label>
                      <Input
                        value={configuration.default_channel || ''}
                        onChange={(e) => handleConfigurationChange('default_channel', e.target.value)}
                        placeholder="#general"
                      />
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={configuration.notifications_enabled || false}
                        onCheckedChange={(checked) => handleConfigurationChange('notifications_enabled', checked)}
                      />
                      <Label>Enable notifications</Label>
                    </div>
                    <div className="space-y-2">
                      <Label>Notification Frequency</Label>
                      <Select value={configuration.notification_frequency || 'immediate'} onValueChange={(value) => handleConfigurationChange('notification_frequency', value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select frequency" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="immediate">Immediate</SelectItem>
                          <SelectItem value="hourly">Hourly</SelectItem>
                          <SelectItem value="daily">Daily</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              )}

              {tool?.name === 'AWS S3' && (
                <Card className="bg-secondary/30 border-border/30">
                  <CardHeader>
                    <CardTitle className="text-base">AWS S3 Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Default Bucket</Label>
                      <Input
                        value={configuration.default_bucket || ''}
                        onChange={(e) => handleConfigurationChange('default_bucket', e.target.value)}
                        placeholder="my-app-bucket"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Region</Label>
                      <Select value={configuration.region || 'us-east-1'} onValueChange={(value) => handleConfigurationChange('region', value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select region" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="us-east-1">US East (N. Virginia)</SelectItem>
                          <SelectItem value="us-west-2">US West (Oregon)</SelectItem>
                          <SelectItem value="eu-west-1">Europe (Ireland)</SelectItem>
                          <SelectItem value="ap-southeast-1">Asia Pacific (Singapore)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={configuration.encryption_enabled || false}
                        onCheckedChange={(checked) => handleConfigurationChange('encryption_enabled', checked)}
                      />
                      <Label>Enable server-side encryption</Label>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Generic Configuration for other tools */}
              {!['GitHub', 'Slack', 'AWS S3'].includes(tool?.name || '') && (
                <Card className="bg-secondary/30 border-border/30">
                  <CardHeader>
                    <CardTitle className="text-base">General Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Custom Settings</Label>
                      <Textarea
                        value={JSON.stringify(configuration, null, 2)}
                        onChange={(e) => {
                          try {
                            const parsed = JSON.parse(e.target.value)
                            setConfiguration(parsed)
                          } catch (error) {
                            // Invalid JSON, ignore
                          }
                        }}
                        placeholder="Enter configuration as JSON"
                        rows={6}
                      />
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="security" className="space-y-6">
              {/* Agent Permissions */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Shield className="w-5 h-5" />
                    <span>Agent Permissions</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                      This tool can be accessed by the following agent types:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {tool?.permissions?.map((permission) => (
                        <Badge key={permission} className="bg-green-500/10 text-green-400 border-green-500/20">
                          {permission.replace('_', ' ').toUpperCase()}
                        </Badge>
                      )) || []}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Security Settings */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base">Security Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Credential Encryption</p>
                      <p className="text-sm text-muted-foreground">All credentials are encrypted at rest</p>
                    </div>
                    <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                      Enabled
                    </Badge>
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Access Logging</p>
                      <p className="text-sm text-muted-foreground">Log all tool access attempts</p>
                    </div>
                    <Switch
                      checked={configuration.access_logging !== false}
                      onCheckedChange={(checked) => handleConfigurationChange('access_logging', checked)}
                    />
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Rate Limiting</p>
                      <p className="text-sm text-muted-foreground">Limit API requests per minute</p>
                    </div>
                    <Switch
                      checked={configuration.rate_limiting !== false}
                      onCheckedChange={(checked) => handleConfigurationChange('rate_limiting', checked)}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="testing" className="space-y-6">
              {/* Connection Test */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <TestTube className="w-5 h-5" />
                    <span>Test Connection</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    Test your configuration to ensure the tool is properly connected and configured.
                  </p>
                  
                  <Button
                    onClick={handleTestConnection}
                    disabled={testConnection.testing}
                    className="w-full"
                  >
                    {testConnection.testing ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Testing Connection...
                      </>
                    ) : (
                      <>
                        <TestTube className="w-4 h-4 mr-2" />
                        Test Connection
                      </>
                    )}
                  </Button>

                  {testConnection.status && (
                    <Alert className={testConnection.status === 'success' 
                      ? 'border-green-500/50 bg-green-500/10' 
                      : 'border-red-500/50 bg-red-500/10'
                    }>
                      {testConnection.status === 'success' ? (
                        <CheckCircle className="h-4 w-4 text-green-400" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-red-400" />
                      )}
                      <AlertDescription className={testConnection.status === 'success' ? 'text-green-400' : 'text-red-400'}>
                        {testConnection.message}
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              {/* Configuration Preview */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base">Configuration Preview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="bg-background/50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Label className="text-sm font-medium">Final Configuration</Label>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => copyToClipboard(JSON.stringify({ ...configuration, credentials: '***' }, null, 2))}
                        >
                          <Copy className="w-4 h-4 mr-1" />
                          Copy
                        </Button>
                      </div>
                      <pre className="text-xs text-muted-foreground overflow-auto">
                        {JSON.stringify({ 
                          environment: selectedEnvironment,
                          ...configuration,
                          credentials: Object.keys(credentials).reduce((acc, key) => ({
                            ...acc,
                            [key]: credentials[key] ? '***CONFIGURED***' : 'NOT SET'
                          }), {})
                        }, null, 2)}
                      </pre>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-4 border-t border-border/30">
          <div className="text-sm text-muted-foreground">
            Environment: <Badge className={environmentColors[selectedEnvironment as keyof typeof environmentColors]}>
              {selectedEnvironment}
            </Badge>
          </div>
          <div className="flex space-x-3">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              onClick={handleSave}
              disabled={saving}
              className="icon-gradient"
            >
              {saving ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save Configuration
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
