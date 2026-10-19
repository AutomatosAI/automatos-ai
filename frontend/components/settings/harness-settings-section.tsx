/**
 * Settings > Orchestrator, section 4: HARNESS — Self-Optimizing Organization Loop
 * (PRD-121) — the Enable switch, the run schedule and the approval mode. Moved out of
 * SystemLLMSettingsTab.tsx (PRD-263 PR-1E-a) so each part fits the component length
 * rule; markup unchanged.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChevronDown, Shield, Calendar } from 'lucide-react'
import { HarnessConfig, OrchestratorConfig } from './orchestrator-settings-config'

type HarnessChange = (key: string, value: string | boolean) => void

interface HarnessSettingsSectionProps {
  orchConfig: OrchestratorConfig | null
  onHarnessChange: HarnessChange
}

export function HarnessSettingsSection({ orchConfig, onHarnessChange }: HarnessSettingsSectionProps) {
  const [harnessOpen, setHarnessOpen] = useState(false)

  return (
    <Collapsible open={harnessOpen} onOpenChange={setHarnessOpen}>
      <Card>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                <div>
                  <CardTitle>HARNESS</CardTitle>
                  <CardDescription>
                    Self-optimizing loop that tunes agent configurations based on performance data
                  </CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {orchConfig?.harness?.enabled && (
                  <Badge variant="default" className="text-xs">Active</Badge>
                )}
                <ChevronDown className={`h-5 w-5 transition-transform ${harnessOpen ? 'rotate-180' : ''}`} />
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-6">
            {orchConfig ? (
              <>
                {/* Description */}
                <div className="rounded-lg bg-secondary/30 p-4 text-sm text-muted-foreground">
                  <p className="mb-2">
                    <strong className="text-foreground">HARNESS</strong> (Holistic Agent Review, Normalization, Evaluation &amp; Self-Shaping)
                    periodically collects org-wide metrics, diagnoses regressions, and prescribes configuration
                    changes to keep your agent fleet running optimally.
                  </p>
                  <p>
                    Safe changes (low risk) can be applied automatically, or you can require manual approval
                    for every change. Risky changes are always queued as board tasks for your review.
                  </p>
                </div>

                {/* Enable Toggle */}
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Enable HARNESS</Label>
                    <p className="text-xs text-muted-foreground">
                      When disabled, no automatic optimization runs will occur
                    </p>
                  </div>
                  <Switch
                    checked={orchConfig.harness?.enabled ?? false}
                    onCheckedChange={(checked) => onHarnessChange('enabled', checked)}
                  />
                </div>

                {orchConfig.harness?.enabled && (
                  <HarnessScheduleFields harness={orchConfig.harness} onHarnessChange={onHarnessChange} />
                )}
              </>
            ) : null}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

interface HarnessScheduleFieldsProps {
  harness: HarnessConfig
  onHarnessChange: HarnessChange
}

function HarnessScheduleFields({ harness, onHarnessChange }: HarnessScheduleFieldsProps) {
  return (
    <>
      {/* Schedule & Mode */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            Run Schedule
          </Label>
          <Select
            value={harness.schedule || 'weekly'}
            onValueChange={(v) => onHarnessChange('schedule', v)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="weekly">Weekly (Sunday 2 AM UTC)</SelectItem>
              <SelectItem value="biweekly">Biweekly</SelectItem>
              <SelectItem value="monthly">Monthly</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            How often HARNESS analyzes your organization and proposes changes
          </p>
        </div>

        <div className="space-y-2">
          <Label className="flex items-center gap-1">
            <Shield className="h-3 w-3" />
            Approval Mode
          </Label>
          <Select
            value={harness.mode || 'full_auto'}
            onValueChange={(v) => onHarnessChange('mode', v)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="full_auto">Full Auto — low-risk changes applied automatically</SelectItem>
              <SelectItem value="manual">Manual — all changes require your approval</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {harness.mode === 'manual'
              ? 'Every proposed change will be queued as a board task for your review'
              : 'Changes with risk score 1-2 are applied automatically; risk 3+ queued for review'}
          </p>
        </div>
      </div>
    </>
  )
}
