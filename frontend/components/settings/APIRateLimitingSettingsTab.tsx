/**
 * API Rate Limiting Settings Tab Component
 * ========================================
 * 
 * Manages API rate limiting configuration.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Save, RotateCcw, Shield, Settings, Zap } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface APIRateLimitingSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function APIRateLimitingSettingsTab({ 
  settings, 
  onSave, 
  saving, 
  onReset 
}: APIRateLimitingSettingsTabProps) {
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

  return (
    <div className="space-y-6">
      {/* Rate Limiting Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Rate Limiting Configuration
          </CardTitle>
          <CardDescription>
            Configure API rate limiting to prevent abuse and ensure fair usage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable Rate Limiting</Label>
                <p className="text-sm text-muted-foreground">
                  Enable rate limiting for all API endpoints
                </p>
              </div>
              <Switch
                checked={formData.enabled === 'true'}
                onCheckedChange={(checked) => handleInputChange('enabled', checked.toString())}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="requests_per_window">Requests per Window</Label>
                <Input
                  id="requests_per_window"
                  type="number"
                  min="1"
                  max="10000"
                  value={formData.requests_per_window || ''}
                  onChange={(e) => handleInputChange('requests_per_window', e.target.value)}
                  placeholder="100"
                />
                <p className="text-xs text-muted-foreground">
                  Maximum requests allowed per time window
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="window_seconds">Window Duration (seconds)</Label>
                <Input
                  id="window_seconds"
                  type="number"
                  min="1"
                  max="3600"
                  value={formData.window_seconds || ''}
                  onChange={(e) => handleInputChange('window_seconds', e.target.value)}
                  placeholder="60"
                />
                <p className="text-xs text-muted-foreground">
                  Time window for rate limiting
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Rate Limiting */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Advanced Rate Limiting
          </CardTitle>
          <CardDescription>
            Fine-tune rate limiting behavior and exceptions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="burst_limit">Burst Limit</Label>
              <Input
                id="burst_limit"
                type="number"
                min="1"
                max="1000"
                value={formData.burst_limit || '200'}
                onChange={(e) => handleInputChange('burst_limit', e.target.value)}
                placeholder="200"
              />
              <p className="text-xs text-muted-foreground">
                Maximum burst requests allowed
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="recovery_time">Recovery Time (seconds)</Label>
              <Input
                id="recovery_time"
                type="number"
                min="1"
                max="3600"
                value={formData.recovery_time || '300'}
                onChange={(e) => handleInputChange('recovery_time', e.target.value)}
                placeholder="300"
              />
              <p className="text-xs text-muted-foreground">
                Time to recover from rate limit
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="skip_successful_requests">Skip Successful Requests</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={formData.skip_successful_requests === 'true'}
                  onCheckedChange={(checked) => handleInputChange('skip_successful_requests', checked.toString())}
                />
                <Label>Don't count successful requests</Label>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="skip_failed_requests">Skip Failed Requests</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={formData.skip_failed_requests === 'true'}
                  onCheckedChange={(checked) => handleInputChange('skip_failed_requests', checked.toString())}
                />
                <Label>Don't count failed requests</Label>
              </div>
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
            Optimize rate limiting performance and storage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="storage_backend">Storage Backend</Label>
              <Input
                id="storage_backend"
                value={formData.storage_backend || 'redis'}
                onChange={(e) => handleInputChange('storage_backend', e.target.value)}
                placeholder="redis"
              />
              <p className="text-xs text-muted-foreground">
                Storage backend for rate limiting (redis, memory)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="cleanup_interval">Cleanup Interval (seconds)</Label>
              <Input
                id="cleanup_interval"
                type="number"
                min="60"
                max="3600"
                value={formData.cleanup_interval || '300'}
                onChange={(e) => handleInputChange('cleanup_interval', e.target.value)}
                placeholder="300"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max_keys">Max Keys</Label>
              <Input
                id="max_keys"
                type="number"
                min="1000"
                max="1000000"
                value={formData.max_keys || '100000'}
                onChange={(e) => handleInputChange('max_keys', e.target.value)}
                placeholder="100000"
              />
              <p className="text-xs text-muted-foreground">
                Maximum number of rate limit keys to store
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="key_expiry">Key Expiry (seconds)</Label>
              <Input
                id="key_expiry"
                type="number"
                min="60"
                max="86400"
                value={formData.key_expiry || '3600'}
                onChange={(e) => handleInputChange('key_expiry', e.target.value)}
                placeholder="3600"
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
