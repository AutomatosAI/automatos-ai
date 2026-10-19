/**
 * Settings > Orchestrator, section 4: HARNESS (PRD-121; PRD-263 US-017).
 *
 * The mode is the only HARNESS control: one select, always shown, with no on/off
 * switch and no schedule (the weekly run's day and hour are an installation
 * setting). In P1 it offers the two modes that exist: Report only and Disabled.
 * US-054 adds Propose when proposals can exist. A stored value this list does
 * not offer (a legacy mode, or Propose before US-054) shows as Report only, which
 * is how HARNESS runs it.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChevronDown, Shield } from 'lucide-react'
import {
  HARNESS_HELP_TEXT,
  HARNESS_MODE_OPTIONS,
  HarnessConfig,
  OrchestratorConfig,
  harnessModeOption,
} from './orchestrator-settings-config'

type HarnessChange = (key: string, value: string) => void

interface HarnessSettingsSectionProps {
  orchConfig: OrchestratorConfig | null
  onHarnessChange: HarnessChange
}

export function HarnessSettingsSection({ orchConfig, onHarnessChange }: HarnessSettingsSectionProps) {
  const [harnessOpen, setHarnessOpen] = useState(false)
  const harness = orchConfig?.harness
  const option = harnessModeOption(harness?.mode)

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
                    A weekly review of your agents, from their real Tasks and runs
                  </CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {harness && (
                  <Badge variant="outline" className="text-xs">{option.label}</Badge>
                )}
                <ChevronDown className={`h-5 w-5 transition-transform ${harnessOpen ? 'rotate-180' : ''}`} />
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-6">
            {harness ? (
              <>
                {/* Description: what the chosen mode does */}
                <div className="rounded-lg bg-secondary/30 p-4 text-sm text-muted-foreground">
                  <p>
                    <strong className="text-foreground">{option.label}:</strong> {option.description}
                  </p>
                </div>

                <HarnessModeField harness={harness} onHarnessChange={onHarnessChange} />
              </>
            ) : null}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

interface HarnessModeFieldProps {
  harness: HarnessConfig
  onHarnessChange: HarnessChange
}

/** Mode: the only HARNESS control. */
function HarnessModeField({ harness, onHarnessChange }: HarnessModeFieldProps) {
  return (
    <div className="space-y-2">
      <Label className="flex items-center gap-1">
        <Shield className="h-3 w-3" />
        Mode
      </Label>
      <Select
        value={harnessModeOption(harness.mode).value}
        onValueChange={(v) => onHarnessChange('mode', v)}
      >
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {HARNESS_MODE_OPTIONS.map((item) => (
            <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">{HARNESS_HELP_TEXT}</p>
    </div>
  )
}
