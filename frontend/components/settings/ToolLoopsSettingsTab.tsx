/**
 * Tool Loops Settings Tab
 * ========================
 *
 * Surfaces the four tool-loop iteration limits that govern how many tool
 * calls each agent surface gets per turn:
 *   - chatbot              — Auto's per-message budget (mid-conversation)
 *   - recipe               — recipe step default
 *   - agent_heartbeat      — heartbeat tick budget
 *   - coordinator          — mission coordinator per-task budget
 *
 * Each category renders as a card with integer inputs driven by the
 * setting's `description` and `validation_rules.min/max` from the DB.
 * Per-category save and reset buttons isolate changes — no global save.
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Save, RotateCcw, MessageSquare, ListChecks, Activity, Workflow } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface ToolLoopsSettingsTabProps {
  chatbotSettings: SystemSetting[]
  recipeSettings: SystemSetting[]
  heartbeatSettings: SystemSetting[]
  coordinatorSettings: SystemSetting[]
  onSaveChatbot: (updates: Record<string, string>) => void
  onSaveRecipe: (updates: Record<string, string>) => void
  onSaveHeartbeat: (updates: Record<string, string>) => void
  onSaveCoordinator: (updates: Record<string, string>) => void
  onResetChatbot: () => void
  onResetRecipe: () => void
  onResetHeartbeat: () => void
  onResetCoordinator: () => void
  saving: boolean
}

interface CategoryCardProps {
  title: string
  description: string
  icon: LucideIcon
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  onReset: () => void
  saving: boolean
}

function humanizeKey(key: string): string {
  return key
    .split('_')
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function CategoryCard({
  title,
  description,
  icon: Icon,
  settings,
  onSave,
  onReset,
  saving,
}: CategoryCardProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})

  useEffect(() => {
    const initial: Record<string, string> = {}
    settings.forEach(s => {
      initial[s.key] = s.value !== null && s.value !== undefined
        ? s.value
        : (s.default_value || '')
    })
    setFormData(initial)
  }, [settings])

  const handleChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleReset = () => {
    const defaults: Record<string, string> = {}
    settings.forEach(s => { defaults[s.key] = s.default_value || '' })
    setFormData(defaults)
    onReset()
  }

  if (settings.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icon className="h-5 w-5" />
            {title}
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No settings found for this category. Run{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">
              python -m core.seeds.seed_system_settings
            </code>{' '}
            to seed defaults.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className="h-5 w-5" />
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {settings.map(setting => {
            const min = setting.validation_rules?.min as number | undefined
            const max = setting.validation_rules?.max as number | undefined
            return (
              <div key={setting.id} className="space-y-2">
                <Label htmlFor={`${setting.category}-${setting.key}`}>
                  {humanizeKey(setting.key)}
                </Label>
                <Input
                  id={`${setting.category}-${setting.key}`}
                  type={setting.value_type === 'number' ? 'number' : 'text'}
                  min={min}
                  max={max}
                  value={formData[setting.key] ?? ''}
                  onChange={e => handleChange(setting.key, e.target.value)}
                  placeholder={setting.default_value || ''}
                />
                {setting.description && (
                  <p className="text-xs text-muted-foreground">
                    {setting.description}
                  </p>
                )}
              </div>
            )
          })}
        </div>

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" size="sm" onClick={handleReset} disabled={saving}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button size="sm" onClick={() => onSave(formData)} disabled={saving}>
            <Save className="h-4 w-4 mr-2" />
            Save Changes
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export default function ToolLoopsSettingsTab({
  chatbotSettings,
  recipeSettings,
  heartbeatSettings,
  coordinatorSettings,
  onSaveChatbot,
  onSaveRecipe,
  onSaveHeartbeat,
  onSaveCoordinator,
  onResetChatbot,
  onResetRecipe,
  onResetHeartbeat,
  onResetCoordinator,
  saving,
}: ToolLoopsSettingsTabProps) {
  return (
    <div className="space-y-6">
      <div className="rounded-md border bg-muted/40 p-4 text-sm text-muted-foreground">
        Each surface below caps how many tool-call turns its agents get per
        iteration. Higher values let agents chain more steps; lower values fail
        faster. These were previously hardcoded — bumping the chatbot ceiling
        is the canonical fix for Auto saying &ldquo;I&rsquo;m blocked&rdquo;
        mid-conversation.
      </div>

      <CategoryCard
        title="Chatbot — Auto Conversation"
        description="How many tool calls Auto can chain in a single user reply, plus the retry budgets for transient routing/parameter failures."
        icon={MessageSquare}
        settings={chatbotSettings}
        onSave={onSaveChatbot}
        onReset={onResetChatbot}
        saving={saving}
      />

      <CategoryCard
        title="Recipe Steps"
        description="Default tool-call budget per recipe step. Per-step (step.max_iterations) and per-agent (agent.configuration.max_iterations) overrides take precedence."
        icon={ListChecks}
        settings={recipeSettings}
        onSave={onSaveRecipe}
        onReset={onResetRecipe}
        saving={saving}
      />

      <CategoryCard
        title="Agent Heartbeat"
        description="Tool-call budget for each scheduled heartbeat tick — short health-check runs that catch agent drift between conversations."
        icon={Activity}
        settings={heartbeatSettings}
        onSave={onSaveHeartbeat}
        onReset={onResetHeartbeat}
        saving={saving}
      />

      <CategoryCard
        title="Mission Coordinator"
        description="Per-task tool-call budget when the coordinator runs orchestrated mission tasks concurrently."
        icon={Workflow}
        settings={coordinatorSettings}
        onSave={onSaveCoordinator}
        onReset={onResetCoordinator}
        saving={saving}
      />
    </div>
  )
}
