/**
 * Backend API Keys Settings Tab Component
 * =======================================
 * 
 * Manages backend API authentication keys and configuration.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw, Key, Shield, Settings } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface BackendAPIKeysSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function BackendAPIKeysSettingsTab({ 
  settings, 
  onSave, 
  saving, 
  onReset 
}: BackendAPIKeysSettingsTabProps) {
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
      {/* API Authentication */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            API Authentication
          </CardTitle>
          <CardDescription>
            Configure backend API authentication keys and security settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="api_key">API Key</Label>
              <Input
                id="api_key"
                type="password"
                value={formData.api_key || ''}
                onChange={(e) => handleInputChange('api_key', e.target.value)}
                placeholder="test_api_key_for_backend_validation_2025"
              />
              {getSetting('api_key')?.is_sensitive && (
                <Badge variant="destructive" className="text-xs">Sensitive</Badge>
              )}
              {getSetting('api_key')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
              <p className="text-xs text-muted-foreground">
                Main API key for backend authentication
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_port">API Port</Label>
              <Input
                id="api_port"
                type="number"
                min="1"
                max="65535"
                value={formData.api_port || ''}
                onChange={(e) => handleInputChange('api_port', e.target.value)}
                placeholder="8000"
              />
              {getSetting('api_port')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_url">API URL</Label>
              <Input
                id="api_url"
                value={formData.api_url || ''}
                onChange={(e) => handleInputChange('api_url', e.target.value)}
                placeholder="api.automatos.app"
              />
              {getSetting('api_url')?.is_required && (
                <Badge variant="destructive" className="text-xs">Required</Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Security Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Security Settings
          </CardTitle>
          <CardDescription>
            Configure API security and access control
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="api_key_length">API Key Length</Label>
              <Input
                id="api_key_length"
                type="number"
                min="16"
                max="128"
                value={formData.api_key_length || '32'}
                onChange={(e) => handleInputChange('api_key_length', e.target.value)}
                placeholder="32"
              />
              <p className="text-xs text-muted-foreground">
                Minimum length for generated API keys
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_key_expiry">API Key Expiry (days)</Label>
              <Input
                id="api_key_expiry"
                type="number"
                min="1"
                max="365"
                value={formData.api_key_expiry || '90'}
                onChange={(e) => handleInputChange('api_key_expiry', e.target.value)}
                placeholder="90"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max_api_keys_per_user">Max API Keys per User</Label>
              <Input
                id="max_api_keys_per_user"
                type="number"
                min="1"
                max="10"
                value={formData.max_api_keys_per_user || '5'}
                onChange={(e) => handleInputChange('max_api_keys_per_user', e.target.value)}
                placeholder="5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_key_prefix">API Key Prefix</Label>
              <Input
                id="api_key_prefix"
                value={formData.api_key_prefix || 'ak_'}
                onChange={(e) => handleInputChange('api_key_prefix', e.target.value)}
                placeholder="ak_"
              />
              <p className="text-xs text-muted-foreground">
                Prefix for generated API keys
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Advanced Settings
          </CardTitle>
          <CardDescription>
            Configure advanced API behavior and monitoring
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="api_timeout">API Timeout (seconds)</Label>
              <Input
                id="api_timeout"
                type="number"
                min="5"
                max="300"
                value={formData.api_timeout || '30'}
                onChange={(e) => handleInputChange('api_timeout', e.target.value)}
                placeholder="30"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_retry_attempts">Retry Attempts</Label>
              <Input
                id="api_retry_attempts"
                type="number"
                min="0"
                max="5"
                value={formData.api_retry_attempts || '3'}
                onChange={(e) => handleInputChange('api_retry_attempts', e.target.value)}
                placeholder="3"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_rate_limit_per_key">Rate Limit per Key</Label>
              <Input
                id="api_rate_limit_per_key"
                type="number"
                min="1"
                max="10000"
                value={formData.api_rate_limit_per_key || '1000'}
                onChange={(e) => handleInputChange('api_rate_limit_per_key', e.target.value)}
                placeholder="1000"
              />
              <p className="text-xs text-muted-foreground">
                Requests per hour per API key
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="api_monitoring_enabled">Enable API Monitoring</Label>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.api_monitoring_enabled === 'true'}
                  onChange={(e) => handleInputChange('api_monitoring_enabled', e.target.checked.toString())}
                  className="rounded"
                />
                <Label>Monitor API usage and performance</Label>
              </div>
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
