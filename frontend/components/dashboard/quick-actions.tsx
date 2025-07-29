
'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Plus, Upload, Play, Settings, Save, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import Link from 'next/link'

const actions = [
  {
    label: 'Create Agent',
    href: '/agents?action=create',
    icon: Plus,
    description: 'Add a new AI agent to your system',
    type: 'link'
  },
  {
    label: 'Upload Documents',
    href: '/documents?action=upload',
    icon: Upload,
    description: 'Add documents to knowledge base',
    type: 'link'
  },
  {
    label: 'Start Workflow',
    href: '/workflows?action=create',
    icon: Play,
    description: 'Create and run a new workflow',
    type: 'link'
  },
  {
    label: 'System Settings',
    href: '/settings',
    icon: Settings,
    description: 'Configure system parameters',
    type: 'modal'
  }
]

export function QuickActions() {
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [settings, setSettings] = useState({
    // System Configuration
    systemName: 'Automotas AI',
    maxConcurrentAgents: 10,
    defaultTimeout: 30,
    apiRetryLimit: 3,
    
    // Performance Settings
    enableCaching: true,
    cacheExpiry: 3600,
    maxMemoryUsage: 8,
    enableLogging: true,
    logLevel: 'info',
    
    // Security Settings
    enableEncryption: true,
    sessionTimeout: 24,
    requireAuthentication: true,
    allowedOrigins: 'http://localhost:3000,http://localhost:8002',
    
    // AI Model Settings
    defaultModel: 'gpt-4o-mini',
    maxTokens: 4000,
    temperature: 0.7,
    enableStreamingResponses: true,
    
    // Notification Settings
    enableEmailNotifications: false,
    enableSlackIntegration: false,
    alertThresholds: {
      errorRate: 5,
      responseTime: 5000,
      memoryUsage: 90
    }
  })

  const handleOpenSettings = () => {
    setShowSettingsModal(true)
  }

  const handleSaveSettings = async () => {
    setIsSaving(true)
    try {
      const response = await fetch('http://localhost:8002/api/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      })

      if (!response.ok) {
        throw new Error('Failed to save settings')
      }

      // Close modal after successful save
      setTimeout(() => {
        setShowSettingsModal(false)
      }, 1000)
    } catch (error) {
      console.error('Error saving settings:', error)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <>
      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {actions.map((action, index) => (
            <motion.div
              key={action.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {action.type === 'link' ? (
                <Link href={action.href}>
                  <Button
                    variant="outline"
                    className="w-full h-auto p-4 flex flex-col items-center space-y-2 hover:bg-secondary/50 hover:border-primary/30 transition-all duration-200 group"
                  >
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                      <action.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-sm">{action.label}</div>
                      <div className="text-xs text-muted-foreground">{action.description}</div>
                    </div>
                  </Button>
                </Link>
              ) : (
                <Button
                  variant="outline"
                  className="w-full h-auto p-4 flex flex-col items-center space-y-2 hover:bg-secondary/50 hover:border-primary/30 transition-all duration-200 group"
                  onClick={handleOpenSettings}
                >
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                    <action.icon className="w-5 h-5 text-primary" />
                  </div>
                  <div className="text-center">
                    <div className="font-medium text-sm">{action.label}</div>
                    <div className="text-xs text-muted-foreground">{action.description}</div>
                  </div>
                </Button>
              )}
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* System Settings Modal */}
      <Dialog open={showSettingsModal} onOpenChange={setShowSettingsModal}>
        <DialogContent className="glass-card max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between">
              <span>System Settings</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowSettingsModal(false)}
              >
                <X className="w-4 h-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>

          <Tabs defaultValue="system" className="space-y-6">
            <TabsList className="grid w-full grid-cols-5 bg-secondary/50">
              <TabsTrigger value="system">System</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="security">Security</TabsTrigger>
              <TabsTrigger value="ai">AI Models</TabsTrigger>
              <TabsTrigger value="notifications">Notifications</TabsTrigger>
            </TabsList>

            {/* System Configuration */}
            <TabsContent value="system" className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>General Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="systemName">System Name</Label>
                      <Input
                        id="systemName"
                        value={settings.systemName}
                        onChange={(e) => setSettings({...settings, systemName: e.target.value})}
                        className="bg-secondary/50 border-secondary"
                      />
                    </div>
                    <div>
                      <Label htmlFor="maxAgents">Max Concurrent Agents</Label>
                      <Input
                        id="maxAgents"
                        type="number"
                        value={settings.maxConcurrentAgents}
                        onChange={(e) => setSettings({...settings, maxConcurrentAgents: parseInt(e.target.value) || 10})}
                        className="bg-secondary/50 border-secondary"
                      />
                    </div>
                    <div>
                      <Label htmlFor="defaultTimeout">Default Timeout (minutes)</Label>
                      <Input
                        id="defaultTimeout"
                        type="number"
                        value={settings.defaultTimeout}
                        onChange={(e) => setSettings({...settings, defaultTimeout: parseInt(e.target.value) || 30})}
                        className="bg-secondary/50 border-secondary"
                      />
                    </div>
                    <div>
                      <Label htmlFor="retryLimit">API Retry Limit</Label>
                      <Input
                        id="retryLimit"
                        type="number"
                        value={settings.apiRetryLimit}
                        onChange={(e) => setSettings({...settings, apiRetryLimit: parseInt(e.target.value) || 3})}
                        className="bg-secondary/50 border-secondary"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Performance Settings */}
            <TabsContent value="performance" className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Performance Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="enableCaching">Enable Caching</Label>
                        <Switch
                          id="enableCaching"
                          checked={settings.enableCaching}
                          onCheckedChange={(checked) => setSettings({...settings, enableCaching: checked})}
                        />
                      </div>
                      <div>
                        <Label htmlFor="cacheExpiry">Cache Expiry (seconds)</Label>
                        <Input
                          id="cacheExpiry"
                          type="number"
                          value={settings.cacheExpiry}
                          onChange={(e) => setSettings({...settings, cacheExpiry: parseInt(e.target.value) || 3600})}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                      <div>
                        <Label htmlFor="maxMemory">Max Memory Usage (GB)</Label>
                        <Input
                          id="maxMemory"
                          type="number"
                          value={settings.maxMemoryUsage}
                          onChange={(e) => setSettings({...settings, maxMemoryUsage: parseInt(e.target.value) || 8})}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="enableLogging">Enable Logging</Label>
                        <Switch
                          id="enableLogging"
                          checked={settings.enableLogging}
                          onCheckedChange={(checked) => setSettings({...settings, enableLogging: checked})}
                        />
                      </div>
                      <div>
                        <Label htmlFor="logLevel">Log Level</Label>
                        <Select 
                          value={settings.logLevel} 
                          onValueChange={(value) => setSettings({...settings, logLevel: value})}
                        >
                          <SelectTrigger className="bg-secondary/50 border-secondary">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="debug">Debug</SelectItem>
                            <SelectItem value="info">Info</SelectItem>
                            <SelectItem value="warn">Warning</SelectItem>
                            <SelectItem value="error">Error</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Security Settings */}
            <TabsContent value="security" className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Security Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="enableEncryption">Enable Encryption</Label>
                        <Switch
                          id="enableEncryption"
                          checked={settings.enableEncryption}
                          onCheckedChange={(checked) => setSettings({...settings, enableEncryption: checked})}
                        />
                      </div>
                      <div>
                        <Label htmlFor="sessionTimeout">Session Timeout (hours)</Label>
                        <Input
                          id="sessionTimeout"
                          type="number"
                          value={settings.sessionTimeout}
                          onChange={(e) => setSettings({...settings, sessionTimeout: parseInt(e.target.value) || 24})}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="requireAuth">Require Authentication</Label>
                        <Switch
                          id="requireAuth"
                          checked={settings.requireAuthentication}
                          onCheckedChange={(checked) => setSettings({...settings, requireAuthentication: checked})}
                        />
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="allowedOrigins">Allowed Origins</Label>
                        <Textarea
                          id="allowedOrigins"
                          value={settings.allowedOrigins}
                          onChange={(e) => setSettings({...settings, allowedOrigins: e.target.value})}
                          placeholder="https://example.com, https://app.example.com"
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* AI Model Settings */}
            <TabsContent value="ai" className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>AI Model Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="defaultModel">Default Model</Label>
                        <Select 
                          value={settings.defaultModel} 
                          onValueChange={(value) => setSettings({...settings, defaultModel: value})}
                        >
                          <SelectTrigger className="bg-secondary/50 border-secondary">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                            <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                            <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                            <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label htmlFor="maxTokens">Max Tokens</Label>
                        <Input
                          id="maxTokens"
                          type="number"
                          value={settings.maxTokens}
                          onChange={(e) => setSettings({...settings, maxTokens: parseInt(e.target.value) || 4000})}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="temperature">Temperature</Label>
                        <Input
                          id="temperature"
                          type="number"
                          step="0.1"
                          min="0"
                          max="2"
                          value={settings.temperature}
                          onChange={(e) => setSettings({...settings, temperature: parseFloat(e.target.value) || 0.7})}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="enableStreaming">Enable Streaming Responses</Label>
                        <Switch
                          id="enableStreaming"
                          checked={settings.enableStreamingResponses}
                          onCheckedChange={(checked) => setSettings({...settings, enableStreamingResponses: checked})}
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Notification Settings */}
            <TabsContent value="notifications" className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Notification Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="emailNotifications">Email Notifications</Label>
                        <Switch
                          id="emailNotifications"
                          checked={settings.enableEmailNotifications}
                          onCheckedChange={(checked) => setSettings({...settings, enableEmailNotifications: checked})}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="slackIntegration">Slack Integration</Label>
                        <Switch
                          id="slackIntegration"
                          checked={settings.enableSlackIntegration}
                          onCheckedChange={(checked) => setSettings({...settings, enableSlackIntegration: checked})}
                        />
                      </div>
                    </div>
                    <div className="space-y-4">
                      <h4 className="font-semibold">Alert Thresholds</h4>
                      <div>
                        <Label htmlFor="errorRate">Error Rate (%)</Label>
                        <Input
                          id="errorRate"
                          type="number"
                          value={settings.alertThresholds.errorRate}
                          onChange={(e) => setSettings({
                            ...settings,
                            alertThresholds: {
                              ...settings.alertThresholds,
                              errorRate: parseInt(e.target.value) || 5
                            }
                          })}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                      <div>
                        <Label htmlFor="responseTime">Response Time (ms)</Label>
                        <Input
                          id="responseTime"
                          type="number"
                          value={settings.alertThresholds.responseTime}
                          onChange={(e) => setSettings({
                            ...settings,
                            alertThresholds: {
                              ...settings.alertThresholds,
                              responseTime: parseInt(e.target.value) || 5000
                            }
                          })}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                      <div>
                        <Label htmlFor="memoryUsage">Memory Usage (%)</Label>
                        <Input
                          id="memoryUsage"
                          type="number"
                          value={settings.alertThresholds.memoryUsage}
                          onChange={(e) => setSettings({
                            ...settings,
                            alertThresholds: {
                              ...settings.alertThresholds,
                              memoryUsage: parseInt(e.target.value) || 90
                            }
                          })}
                          className="bg-secondary/50 border-secondary"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Modal Actions */}
          <div className="flex justify-end space-x-4 pt-6 border-t border-border/30">
            <Button
              variant="outline"
              onClick={() => setShowSettingsModal(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleSaveSettings}
              disabled={isSaving}
              className="gradient-accent hover:opacity-90"
            >
              <Save className="w-4 h-4 mr-2" />
              {isSaving ? 'Saving...' : 'Save Settings'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
