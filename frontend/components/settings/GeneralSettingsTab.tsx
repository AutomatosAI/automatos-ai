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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw, Globe, Zap, Shield } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import { InlineHelp } from '@/components/ui/help-tooltip'

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
              <Label htmlFor="environment" className="flex items-center gap-1">Environment <InlineHelp id="settings.environment.environment" size="sm" /></Label>
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
              <Label htmlFor="log_level" className="flex items-center gap-1">Log Level <InlineHelp id="settings.environment.log_level" size="sm" /></Label>
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
