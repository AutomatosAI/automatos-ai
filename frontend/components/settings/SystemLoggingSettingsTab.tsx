/**
 * System Logging Settings Tab Component
 * =====================================
 * 
 * Manages system logging configuration.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Save, RotateCcw, FileText, Settings, Zap } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface SystemLoggingSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function SystemLoggingSettingsTab({ 
  settings, 
  onSave, 
  saving, 
  onReset 
}: SystemLoggingSettingsTabProps) {
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
      {/* Logging Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Logging Configuration
          </CardTitle>
          <CardDescription>
            Configure system-wide logging behavior and output
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="log_level">Log Level</Label>
              <Select 
                value={formData.log_level || 'INFO'} 
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

            <div className="space-y-2">
              <Label htmlFor="log_format">Log Format</Label>
              <Select 
                value={formData.log_format || 'json'} 
                onValueChange={(value) => handleInputChange('log_format', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select log format" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="structured">Structured</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_file_path">Log File Path</Label>
              <Input
                id="log_file_path"
                value={formData.log_file_path || '/var/log/automatos/app.log'}
                onChange={(e) => handleInputChange('log_file_path', e.target.value)}
                placeholder="/var/log/automatos/app.log"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_max_size">Max Log File Size (MB)</Label>
              <Input
                id="log_max_size"
                type="number"
                min="1"
                max="1000"
                value={formData.log_max_size || '100'}
                onChange={(e) => handleInputChange('log_max_size', e.target.value)}
                placeholder="100"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Log Rotation Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Log Rotation Settings
          </CardTitle>
          <CardDescription>
            Configure log file rotation and retention
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="log_backup_count">Backup Count</Label>
              <Input
                id="log_backup_count"
                type="number"
                min="1"
                max="30"
                value={formData.log_backup_count || '7'}
                onChange={(e) => handleInputChange('log_backup_count', e.target.value)}
                placeholder="7"
              />
              <p className="text-xs text-muted-foreground">
                Number of backup files to keep
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_rotation_interval">Rotation Interval (hours)</Label>
              <Input
                id="log_rotation_interval"
                type="number"
                min="1"
                max="168"
                value={formData.log_rotation_interval || '24'}
                onChange={(e) => handleInputChange('log_rotation_interval', e.target.value)}
                placeholder="24"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_retention_days">Retention Days</Label>
              <Input
                id="log_retention_days"
                type="number"
                min="1"
                max="365"
                value={formData.log_retention_days || '30'}
                onChange={(e) => handleInputChange('log_retention_days', e.target.value)}
                placeholder="30"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_compress">Compress Old Logs</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={formData.log_compress === 'true'}
                  onCheckedChange={(checked) => handleInputChange('log_compress', checked.toString())}
                />
                <Label>Enable compression</Label>
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
            Optimize logging performance and resource usage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="log_buffer_size">Buffer Size (KB)</Label>
              <Input
                id="log_buffer_size"
                type="number"
                min="1"
                max="1024"
                value={formData.log_buffer_size || '64'}
                onChange={(e) => handleInputChange('log_buffer_size', e.target.value)}
                placeholder="64"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_flush_interval">Flush Interval (seconds)</Label>
              <Input
                id="log_flush_interval"
                type="number"
                min="1"
                max="60"
                value={formData.log_flush_interval || '5'}
                onChange={(e) => handleInputChange('log_flush_interval', e.target.value)}
                placeholder="5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_async">Async Logging</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={formData.log_async === 'true'}
                  onCheckedChange={(checked) => handleInputChange('log_async', checked.toString())}
                />
                <Label>Enable async logging</Label>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="log_queue_size">Queue Size</Label>
              <Input
                id="log_queue_size"
                type="number"
                min="100"
                max="10000"
                value={formData.log_queue_size || '1000'}
                onChange={(e) => handleInputChange('log_queue_size', e.target.value)}
                placeholder="1000"
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
