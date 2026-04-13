/**
 * Generic Category Settings Tab
 * ==============================
 *
 * Auto-renders all system settings for a given category using the
 * value_type and validation_rules from the DB.  Suitable for new
 * categories that don't need a bespoke layout.
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Save, RotateCcw } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface GenericCategorySettingsTabProps {
  title: string
  description: string
  icon?: React.ReactNode
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function GenericCategorySettingsTab({
  title,
  description,
  icon,
  settings,
  onSave,
  saving,
  onReset,
}: GenericCategorySettingsTabProps) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [dirty, setDirty] = useState(false)

  // Initialise local state from props
  useEffect(() => {
    const initial: Record<string, string> = {}
    for (const s of settings) {
      initial[s.key] = s.value ?? s.default_value ?? ''
    }
    setValues(initial)
    setDirty(false)
  }, [settings])

  const handleChange = (key: string, val: string) => {
    setValues(prev => ({ ...prev, [key]: val }))
    setDirty(true)
  }

  const handleSave = () => {
    // Only send changed values
    const updates: Record<string, string> = {}
    for (const s of settings) {
      const current = values[s.key] ?? ''
      const original = s.value ?? s.default_value ?? ''
      if (current !== original) {
        updates[s.key] = current
      }
    }
    if (Object.keys(updates).length > 0) {
      onSave(updates)
      setDirty(false)
    }
  }

  const renderField = (setting: SystemSetting) => {
    const val = values[setting.key] ?? ''
    const rules = setting.validation_rules as Record<string, unknown> | null

    // Dropdown for options
    if (rules?.options && Array.isArray(rules.options)) {
      return (
        <Select value={val} onValueChange={v => handleChange(setting.key, v)}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select..." />
          </SelectTrigger>
          <SelectContent>
            {(rules.options as string[]).map(opt => (
              <SelectItem key={opt} value={opt}>{opt}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      )
    }

    // Boolean toggle
    if (setting.value_type === 'boolean') {
      return (
        <Switch
          checked={val === 'true'}
          onCheckedChange={checked => handleChange(setting.key, checked ? 'true' : 'false')}
        />
      )
    }

    // Number input
    if (setting.value_type === 'number') {
      return (
        <Input
          type="number"
          value={val}
          onChange={e => handleChange(setting.key, e.target.value)}
          min={rules?.min as number | undefined}
          max={rules?.max as number | undefined}
          step={rules?.step as number | undefined}
        />
      )
    }

    // Default: text input
    return (
      <Input
        value={val}
        onChange={e => handleChange(setting.key, e.target.value)}
        type={setting.is_sensitive ? 'password' : 'text'}
      />
    )
  }

  if (settings.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">{icon}{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            No settings found for this category. Run the system settings seed to populate.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">{icon}{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={onReset} disabled={saving}>
              <RotateCcw className="h-3.5 w-3.5 mr-1" /> Reset
            </Button>
            <Button size="sm" onClick={handleSave} disabled={saving || !dirty}>
              <Save className="h-3.5 w-3.5 mr-1" /> Save
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {settings.map(s => (
          <div key={s.key} className="grid grid-cols-3 items-center gap-4">
            <div className="col-span-1">
              <Label className="text-sm font-medium">
                {s.key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </Label>
              {s.description && (
                <p className="text-xs text-muted-foreground mt-0.5">{s.description}</p>
              )}
              <div className="flex gap-1 mt-1">
                {s.is_required && <Badge variant="destructive" className="text-[10px] px-1 py-0">Required</Badge>}
                {s.is_sensitive && <Badge variant="secondary" className="text-[10px] px-1 py-0">Sensitive</Badge>}
              </div>
            </div>
            <div className="col-span-2">
              {renderField(s)}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
