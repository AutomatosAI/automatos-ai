/**
 * System Settings Tab Component
 * =============================
 * 
 * Main component for managing system-wide configuration settings.
 * Replaces .env file with database-backed settings management.
 */

import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Settings, Save, RotateCcw, AlertCircle } from 'lucide-react'
import { toast } from 'sonner'

import { 
  getSettingsByCategory, 
  getSettingsStats, 
  bulkUpdateSettings, 
  resetSettingsToDefaults,
  SystemSettingsByCategory,
  SystemSettingsStats 
} from '@/lib/api/system-settings'

// Import individual tab components
import GeneralSettingsTab from './GeneralSettingsTab'
import SystemLLMSettingsTab from './SystemLLMSettingsTab'
import CodeGraphSettingsTab from './CodeGraphSettingsTab'
import SystemLoggingSettingsTab from './SystemLoggingSettingsTab'
import APIRateLimitingSettingsTab from './APIRateLimitingSettingsTab'
import BackendAPIKeysSettingsTab from './BackendAPIKeysSettingsTab'

interface SystemSettingsTabProps {
  className?: string
}

export default function SystemSettingsTab({ className }: SystemSettingsTabProps) {
  const [settingsByCategory, setSettingsByCategory] = useState<SystemSettingsByCategory[]>([])
  const [stats, setStats] = useState<SystemSettingsStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('general')

  // Load settings data
  const loadSettings = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const [settingsData, statsData] = await Promise.all([
        getSettingsByCategory(),
        getSettingsStats()
      ])
      
      setSettingsByCategory(settingsData)
      setStats(statsData)
    } catch (err) {
      console.error('Failed to load system settings:', err)
      setError(err instanceof Error ? err.message : 'Failed to load settings')
      toast.error('Failed to load system settings')
    } finally {
      setLoading(false)
    }
  }

  // Save settings for a specific category
  const saveCategorySettings = async (category: string, updates: Record<string, string>) => {
    try {
      setSaving(true)
      
      // Find settings for this category
      const categoryData = settingsByCategory.find(cat => cat.category === category)
      if (!categoryData) {
        throw new Error(`Category ${category} not found`)
      }
      
      // Create bulk update request - save ALL settings in category
      // Use formData value if provided, otherwise preserve existing value
      const bulkUpdates = categoryData.settings
        .map(setting => ({
          id: setting.id,
          value: updates[setting.key] !== undefined ? updates[setting.key] : (setting.value || setting.default_value || '')
        }))
      
      if (bulkUpdates.length > 0) {
        await bulkUpdateSettings(bulkUpdates)
        toast.success(`Updated ${bulkUpdates.length} settings in ${category}`)
        
        // Reload settings to get updated values
        await loadSettings()
      } else {
        console.warn(`No settings found for category: ${category}`)
        toast.warning(`No settings found in ${category}`)
      }
    } catch (err) {
      console.error(`Failed to save ${category} settings:`, err)
      toast.error(`Failed to save ${category} settings`)
    } finally {
      setSaving(false)
    }
  }

  // Reset settings to defaults
  const resetToDefaults = async (category?: string) => {
    try {
      setSaving(true)
      
      await resetSettingsToDefaults(category)
      toast.success(category ? `Reset ${category} settings to defaults` : 'Reset all settings to defaults')
      
      // Reload settings
      await loadSettings()
    } catch (err) {
      console.error('Failed to reset settings:', err)
      toast.error('Failed to reset settings to defaults')
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    loadSettings()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading system settings...</span>
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {error}
          <Button 
            variant="outline" 
            size="sm" 
            className="ml-2"
            onClick={loadSettings}
          >
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className={className}>
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Settings className="h-6 w-6" />
              System Settings
            </h2>
            <p className="text-muted-foreground mt-1">
              Manage system-wide configuration settings. These replace environment variables.
            </p>
          </div>
          
          {stats && (
            <div className="flex gap-2">
              <Badge variant="outline">
                {stats.total_settings} Settings
              </Badge>
              <Badge variant="outline">
                {stats.categories.general || 0} Categories
              </Badge>
              {stats.sensitive_settings > 0 && (
                <Badge variant="destructive">
                  {stats.sensitive_settings} Sensitive
                </Badge>
              )}
            </div>
          )}
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="orchestrator_llm">Orchestrator LLM</TabsTrigger>
          <TabsTrigger value="codegraph">CodeGraph</TabsTrigger>
          <TabsTrigger value="system_logging">Logging</TabsTrigger>
          <TabsTrigger value="api_rate_limiting">Rate Limiting</TabsTrigger>
          <TabsTrigger value="backend_api_keys">API Keys</TabsTrigger>
        </TabsList>

        <TabsContent value="general">
          <GeneralSettingsTab
            settings={settingsByCategory.find(cat => cat.category === 'general')?.settings || []}
            onSave={(updates) => saveCategorySettings('general', updates)}
            saving={saving}
            onReset={() => resetToDefaults('general')}
          />
        </TabsContent>

        <TabsContent value="orchestrator_llm">
          <SystemLLMSettingsTab
            settings={settingsByCategory.find(cat => cat.category === 'orchestrator_llm')?.settings || []}
            onSave={(updates) => saveCategorySettings('orchestrator_llm', updates)}
            saving={saving}
            onReset={() => resetToDefaults('orchestrator_llm')}
          />
        </TabsContent>

        <TabsContent value="codegraph">
          <CodeGraphSettingsTab
            settings={settingsByCategory.find(cat => cat.category === 'codegraph')?.settings || []}
            onSave={(updates) => saveCategorySettings('codegraph', updates)}
            saving={saving}
            onReset={() => resetToDefaults('codegraph')}
          />
        </TabsContent>

        <TabsContent value="system_logging">
          <SystemLoggingSettingsTab
            settings={settingsByCategory.find(cat => cat.category === 'system_logging')?.settings || []}
            onSave={(updates) => saveCategorySettings('system_logging', updates)}
            saving={saving}
            onReset={() => resetToDefaults('system_logging')}
          />
        </TabsContent>

        <TabsContent value="api_rate_limiting">
          <APIRateLimitingSettingsTab
            settings={settingsByCategory.find(cat => cat.category === 'api_rate_limiting')?.settings || []}
            onSave={(updates) => saveCategorySettings('api_rate_limiting', updates)}
            saving={saving}
            onReset={() => resetToDefaults('api_rate_limiting')}
          />
        </TabsContent>

        <TabsContent value="backend_api_keys">
          <BackendAPIKeysSettingsTab
            settings={settingsByCategory.find(cat => cat.category === 'backend_api_keys')?.settings || []}
            onSave={(updates) => saveCategorySettings('backend_api_keys', updates)}
            saving={saving}
            onReset={() => resetToDefaults('backend_api_keys')}
          />
        </TabsContent>
      </Tabs>

      {/* Global Actions */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RotateCcw className="h-5 w-5" />
            Global Actions
          </CardTitle>
          <CardDescription>
            Reset all settings to their default values
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button 
            variant="outline" 
            onClick={() => resetToDefaults()}
            disabled={saving}
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <RotateCcw className="h-4 w-4 mr-2" />
            )}
            Reset All Settings to Defaults
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
