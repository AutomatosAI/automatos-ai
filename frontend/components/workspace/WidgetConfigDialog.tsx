'use client'

/**
 * WidgetConfigDialog (US-011)
 *
 * Modal for editing per-widget settings. Shows common options for all widgets
 * and conditional sections for DataWidget and CodeWidget.
 */

import { useCallback, useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type {
  WidgetType,
  WidgetConfig,
  WidgetConfigCommon,
  DataWidgetConfig,
  CodeWidgetConfig,
  WidgetTheme,
} from '@/components/widgets/types'
import {
  DEFAULT_WIDGET_CONFIG,
  DEFAULT_DATA_WIDGET_CONFIG,
  DEFAULT_CODE_WIDGET_CONFIG,
} from '@/components/widgets/types'
import { useWorkspaceStore } from '@/stores/workspace-store'

// ── Helpers ──────────────────────────────────────────────────────────

function isDataConfig(
  cfg: WidgetConfig,
  type: WidgetType,
): cfg is DataWidgetConfig {
  return type === 'data'
}

function isCodeConfig(
  cfg: WidgetConfig,
  type: WidgetType,
): cfg is CodeWidgetConfig {
  return type === 'code'
}

function getDefaultConfig(widgetType: WidgetType): WidgetConfig {
  switch (widgetType) {
    case 'data':
      return { ...DEFAULT_DATA_WIDGET_CONFIG }
    case 'code':
      return { ...DEFAULT_CODE_WIDGET_CONFIG }
    default:
      return { ...DEFAULT_WIDGET_CONFIG }
  }
}

/** Merge persisted config with defaults so every key is present. */
function mergeWithDefaults(
  current: Partial<WidgetConfig> | undefined,
  widgetType: WidgetType,
): WidgetConfig {
  const defaults = getDefaultConfig(widgetType)
  if (!current) return defaults
  return { ...defaults, ...current }
}

// ── Validation ───────────────────────────────────────────────────────

interface ValidationErrors {
  refreshInterval?: string
  rowsPerPage?: string
  fontSize?: string
}

function validate(
  config: WidgetConfig,
  widgetType: WidgetType,
): ValidationErrors {
  const errors: ValidationErrors = {}

  if (config.autoRefresh) {
    const ri = config.refreshInterval
    if (!Number.isFinite(ri) || ri < 5 || ri > 86400) {
      errors.refreshInterval = 'Must be between 5 and 86,400 seconds'
    }
  }

  if (isDataConfig(config, widgetType)) {
    const rpp = config.rowsPerPage
    if (!Number.isFinite(rpp) || rpp < 10 || rpp > 100) {
      errors.rowsPerPage = 'Must be between 10 and 100'
    }
  }

  if (isCodeConfig(config, widgetType)) {
    const fs = config.fontSize
    if (!Number.isFinite(fs) || fs < 10 || fs > 24) {
      errors.fontSize = 'Must be between 10 and 24'
    }
  }

  return errors
}

// ── Props ────────────────────────────────────────────────────────────

export interface WidgetConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  widgetId: string
  widgetType: WidgetType
  currentConfig?: WidgetConfig
}

// ── Component ────────────────────────────────────────────────────────

export function WidgetConfigDialog({
  open,
  onOpenChange,
  widgetId,
  widgetType,
  currentConfig,
}: WidgetConfigDialogProps) {
  const updateWidget = useWorkspaceStore((s) => s.updateWidget)

  // Local draft config (reset whenever the dialog opens)
  const [draft, setDraft] = useState<WidgetConfig>(() =>
    mergeWithDefaults(currentConfig, widgetType),
  )
  const [errors, setErrors] = useState<ValidationErrors>({})

  // Re-seed draft when dialog opens or currentConfig changes
  useEffect(() => {
    if (open) {
      setDraft(mergeWithDefaults(currentConfig, widgetType))
      setErrors({})
    }
  }, [open, currentConfig, widgetType])

  // ── Updaters ─────────────────────────────────────────────────────

  const setField = useCallback(
    <K extends keyof WidgetConfigCommon>(key: K, value: WidgetConfigCommon[K]) => {
      setDraft((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  const setDataField = useCallback(
    <K extends keyof DataWidgetConfig>(key: K, value: DataWidgetConfig[K]) => {
      setDraft((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  const setCodeField = useCallback(
    <K extends keyof CodeWidgetConfig>(key: K, value: CodeWidgetConfig[K]) => {
      setDraft((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  // ── Save / Cancel ────────────────────────────────────────────────

  const handleSave = useCallback(() => {
    const errs = validate(draft, widgetType)
    if (Object.keys(errs).length > 0) {
      setErrors(errs)
      return
    }
    updateWidget(widgetId, { config: draft })
    onOpenChange(false)
  }, [draft, widgetType, widgetId, updateWidget, onOpenChange])

  const handleCancel = useCallback(() => {
    onOpenChange(false)
  }, [onOpenChange])

  // ── Render ───────────────────────────────────────────────────────

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Widget Settings</DialogTitle>
          <DialogDescription>
            Configure display and behavior options for this widget.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-2">
          {/* ── Common Settings ────────────────────────────────── */}
          <fieldset className="space-y-4">
            <legend className="text-sm font-semibold text-foreground">
              Appearance
            </legend>

            {/* Theme */}
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="wc-theme">Theme</Label>
              <Select
                value={draft.theme}
                onValueChange={(v) => setField('theme', v as WidgetTheme)}
              >
                <SelectTrigger id="wc-theme" className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default</SelectItem>
                  <SelectItem value="minimal">Minimal</SelectItem>
                  <SelectItem value="compact">Compact</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Show header */}
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="wc-header">Show header</Label>
              <Switch
                id="wc-header"
                checked={draft.showHeader}
                onCheckedChange={(v) => setField('showHeader', v)}
              />
            </div>

            {/* Show border */}
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="wc-border">Show border</Label>
              <Switch
                id="wc-border"
                checked={draft.showBorder}
                onCheckedChange={(v) => setField('showBorder', v)}
              />
            </div>
          </fieldset>

          {/* ── Refresh Settings ───────────────────────────────── */}
          <fieldset className="space-y-4">
            <legend className="text-sm font-semibold text-foreground">
              Refresh
            </legend>

            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="wc-autorefresh">Auto-refresh</Label>
              <Switch
                id="wc-autorefresh"
                checked={draft.autoRefresh}
                onCheckedChange={(v) => setField('autoRefresh', v)}
              />
            </div>

            {draft.autoRefresh && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-4">
                  <Label htmlFor="wc-interval">Interval (seconds)</Label>
                  <Input
                    id="wc-interval"
                    type="number"
                    min={5}
                    max={86400}
                    className="w-28 text-right"
                    value={draft.refreshInterval}
                    onChange={(e) =>
                      setField('refreshInterval', Number(e.target.value))
                    }
                  />
                </div>
                {errors.refreshInterval && (
                  <p className="text-xs text-destructive">
                    {errors.refreshInterval}
                  </p>
                )}
              </div>
            )}
          </fieldset>

          {/* ── Data Widget Settings ───────────────────────────── */}
          {widgetType === 'data' && isDataConfig(draft, widgetType) && (
            <fieldset className="space-y-4">
              <legend className="text-sm font-semibold text-foreground">
                Data Options
              </legend>

              {/* Rows per page */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-4">
                  <Label htmlFor="wc-rows">Rows per page</Label>
                  <Input
                    id="wc-rows"
                    type="number"
                    min={10}
                    max={100}
                    className="w-28 text-right"
                    value={draft.rowsPerPage}
                    onChange={(e) =>
                      setDataField('rowsPerPage', Number(e.target.value))
                    }
                  />
                </div>
                {errors.rowsPerPage && (
                  <p className="text-xs text-destructive">
                    {errors.rowsPerPage}
                  </p>
                )}
              </div>

              {/* Chart type */}
              <div className="flex items-center justify-between gap-4">
                <Label htmlFor="wc-chart">Chart type</Label>
                <Select
                  value={draft.chartType}
                  onValueChange={(v) =>
                    setDataField('chartType', v as DataWidgetConfig['chartType'])
                  }
                >
                  <SelectTrigger id="wc-chart" className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bar">Bar</SelectItem>
                    <SelectItem value="line">Line</SelectItem>
                    <SelectItem value="pie">Pie</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </fieldset>
          )}

          {/* ── Code Widget Settings ───────────────────────────── */}
          {widgetType === 'code' && isCodeConfig(draft, widgetType) && (
            <fieldset className="space-y-4">
              <legend className="text-sm font-semibold text-foreground">
                Code Options
              </legend>

              {/* Font size */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-4">
                  <Label htmlFor="wc-fontsize">Font size (px)</Label>
                  <Input
                    id="wc-fontsize"
                    type="number"
                    min={10}
                    max={24}
                    className="w-28 text-right"
                    value={draft.fontSize}
                    onChange={(e) =>
                      setCodeField('fontSize', Number(e.target.value))
                    }
                  />
                </div>
                {errors.fontSize && (
                  <p className="text-xs text-destructive">{errors.fontSize}</p>
                )}
              </div>

              {/* Line numbers */}
              <div className="flex items-center justify-between gap-4">
                <Label htmlFor="wc-linenums">Line numbers</Label>
                <Switch
                  id="wc-linenums"
                  checked={draft.lineNumbers}
                  onCheckedChange={(v) => setCodeField('lineNumbers', v)}
                />
              </div>

              {/* Word wrap */}
              <div className="flex items-center justify-between gap-4">
                <Label htmlFor="wc-wordwrap">Word wrap</Label>
                <Switch
                  id="wc-wordwrap"
                  checked={draft.wordWrap}
                  onCheckedChange={(v) => setCodeField('wordWrap', v)}
                />
              </div>
            </fieldset>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
