/**
 * Settings > Orchestrator, section 2: Soul & Personality (PRD-55) — how Auto
 * communicates and behaves, and where it reaches the owner for approvals. Moved out
 * of SystemLLMSettingsTab.tsx (PRD-263 PR-1E-a) so each part fits the component
 * length rule; markup unchanged.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Sparkles, ChevronDown, MessageSquare, Loader2 } from 'lucide-react'
import { InlineHelp } from '@/components/ui/help-tooltip'
import {
  COMMUNICATION_STYLES,
  ConnectedChannel,
  OrchestratorConfig,
  PERSONALITY_PRESETS,
  PROACTIVE_LEVELS,
} from './orchestrator-settings-config'

type OrchChange = (key: string, value: string | boolean) => void

interface OrchestratorSoulSectionProps {
  orchConfig: OrchestratorConfig | null
  orchLoading: boolean
  connectedChannels: ConnectedChannel[]
  onOrchChange: OrchChange
}

export function OrchestratorSoulSection({ orchConfig, orchLoading, connectedChannels, onOrchChange }: OrchestratorSoulSectionProps) {
  const [soulOpen, setSoulOpen] = useState(true)

  return (
    <Collapsible open={soulOpen} onOpenChange={setSoulOpen}>
      <Card>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                <div>
                  <CardTitle>Soul & Personality</CardTitle>
                  <CardDescription>
                    Define how your AI assistant communicates and behaves
                  </CardDescription>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 transition-transform ${soulOpen ? 'rotate-180' : ''}`} />
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-6">
            {orchLoading ? (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading personality settings...
              </div>
            ) : orchConfig ? (
              <>
                <PersonalityFields orchConfig={orchConfig} onOrchChange={onOrchChange} />
                <BehaviourFields orchConfig={orchConfig} connectedChannels={connectedChannels} onOrchChange={onOrchChange} />
              </>
            ) : null}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

function PersonalityFields({ orchConfig, onOrchChange }: { orchConfig: OrchestratorConfig; onOrchChange: OrchChange }) {
  const soulTokenEstimate = orchConfig.custom_soul
    ? Math.ceil(orchConfig.custom_soul.length / 4)
    : 0

  return (
    <>
      {/* Personality Mode */}
      <div className="space-y-3">
        <Label className="flex items-center gap-1">Personality Mode <InlineHelp id="settings.soul.personality_mode" size="sm" /></Label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(PERSONALITY_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => onOrchChange('personality_mode', key)}
              className={`p-3 rounded-lg border text-left transition-colors ${
                orchConfig.personality_mode === key
                  ? 'border-primary bg-primary/10'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              <div className="font-medium text-sm">{preset.label}</div>
              <div className="text-xs text-muted-foreground mt-1">{preset.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Soul Editor (shown when custom mode selected) */}
      {orchConfig.personality_mode === 'custom' && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-1">Custom Soul Prompt <InlineHelp id="settings.soul.custom_soul" size="sm" /></Label>
            <span className="text-xs text-muted-foreground">
              ~{soulTokenEstimate} tokens
            </span>
          </div>
          <Textarea
            value={orchConfig.custom_soul}
            onChange={(e) => onOrchChange('custom_soul', e.target.value)}
            placeholder="Define your AI assistant's personality, tone, and behavior..."
            rows={8}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">
            Markdown supported. This overrides the default personality and is injected into every orchestrator call.
          </p>
        </div>
      )}

      {/* Communication Style */}
      <div className="space-y-3">
        <Label className="flex items-center gap-1">Communication Style <InlineHelp id="settings.soul.communication_style" size="sm" /></Label>
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(COMMUNICATION_STYLES).map(([key, style]) => (
            <button
              key={key}
              onClick={() => onOrchChange('communication_style', key)}
              className={`p-3 rounded-lg border text-left transition-colors ${
                orchConfig.communication_style === key
                  ? 'border-primary bg-primary/10'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              <div className="font-medium text-sm">{style.label}</div>
              <div className="text-xs text-muted-foreground mt-1">{style.description}</div>
            </button>
          ))}
        </div>
      </div>
    </>
  )
}

interface BehaviourFieldsProps {
  orchConfig: OrchestratorConfig
  connectedChannels: ConnectedChannel[]
  onOrchChange: OrchChange
}

function BehaviourFields({ orchConfig, connectedChannels, onOrchChange }: BehaviourFieldsProps) {
  return (
    <>
      {/* Proactive Level */}
      <div className="space-y-3">
        <Label className="flex items-center gap-1">Proactive Behavior Level <InlineHelp id="settings.soul.proactive_level" size="sm" /></Label>
        <Select
          value={orchConfig.proactive_level}
          onValueChange={(v) => onOrchChange('proactive_level', v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(PROACTIVE_LEVELS).map(([key, level]) => (
              <SelectItem key={key} value={key}>
                {level.label} — {level.description}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Controls what the orchestrator does when its heartbeat finds something
        </p>
      </div>

      {/* Preferred Channel — where Auto reaches the owner for approvals */}
      <div className="space-y-3">
        <Label className="flex items-center gap-1">
          <MessageSquare className="h-4 w-4" />
          Preferred Channel for Approvals
        </Label>
        <Select
          value={orchConfig.preferred_channel || 'in_app'}
          onValueChange={(v) => onOrchChange('preferred_channel', v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="in_app">In-App Notification</SelectItem>
            {connectedChannels.map((ch) => (
              <SelectItem key={ch.key} value={ch.platform}>
                {ch.platform.charAt(0).toUpperCase() + ch.platform.slice(1)}
              </SelectItem>
            ))}
            <SelectItem value="webhook">Webhook URL</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Where Auto sends approval requests, decision asks, and urgent escalations.
          A board task is always created as a backup record.
        </p>
      </div>
    </>
  )
}
