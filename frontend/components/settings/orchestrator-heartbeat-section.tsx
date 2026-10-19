/**
 * Settings > Orchestrator, section 3: the orchestrator Heartbeat (PRD-55) — the
 * periodic check-in, its schedule, channel and checklist, Run Now and the last
 * result. Moved out of SystemLLMSettingsTab.tsx (PRD-263 PR-1E-a) so each part fits
 * the component length rule; markup unchanged.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Heart, ChevronDown, Clock, MessageSquare, Loader2, Play } from 'lucide-react'
import { InlineHelp } from '@/components/ui/help-tooltip'
import { ConnectedChannel, HeartbeatConfig, OrchestratorConfig } from './orchestrator-settings-config'

type HeartbeatChange = (key: string, value: string | boolean | number) => void

interface OrchestratorHeartbeatSectionProps {
  orchConfig: OrchestratorConfig | null
  connectedChannels: ConnectedChannel[]
  onHeartbeatChange: HeartbeatChange
  heartbeatRunning: boolean
  lastHeartbeatResult: any
  onRunNow: () => void
}

export function OrchestratorHeartbeatSection(props: OrchestratorHeartbeatSectionProps) {
  const { orchConfig, connectedChannels, onHeartbeatChange, heartbeatRunning, lastHeartbeatResult, onRunNow } = props
  const [heartbeatOpen, setHeartbeatOpen] = useState(false)

  return (
    <Collapsible open={heartbeatOpen} onOpenChange={setHeartbeatOpen}>
      <Card>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Heart className="h-5 w-5" />
                <div>
                  <CardTitle>Orchestrator Heartbeat</CardTitle>
                  <CardDescription>
                    Periodic check-ins — your AI team standup runs automatically
                  </CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {orchConfig?.heartbeat.enabled && (
                  <Badge variant="default" className="text-xs">Active</Badge>
                )}
                <ChevronDown className={`h-5 w-5 transition-transform ${heartbeatOpen ? 'rotate-180' : ''}`} />
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-6">
            {orchConfig ? (
              <>
                {/* Enable Toggle */}
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="flex items-center gap-1">Enable Heartbeat <InlineHelp id="settings.heartbeat.enable" size="sm" /></Label>
                    <p className="text-xs text-muted-foreground">
                      The orchestrator wakes up periodically to check on your agents and tasks
                    </p>
                  </div>
                  <Switch
                    checked={orchConfig.heartbeat.enabled}
                    onCheckedChange={(checked) => onHeartbeatChange('enabled', checked)}
                  />
                </div>

                {orchConfig.heartbeat.enabled && (
                  <>
                    <HeartbeatScheduleFields
                      heartbeat={orchConfig.heartbeat}
                      connectedChannels={connectedChannels}
                      onHeartbeatChange={onHeartbeatChange}
                    />
                    <HeartbeatRunPanel
                      heartbeatRunning={heartbeatRunning}
                      lastHeartbeatResult={lastHeartbeatResult}
                      onRunNow={onRunNow}
                    />
                  </>
                )}
              </>
            ) : null}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

interface HeartbeatScheduleFieldsProps {
  heartbeat: HeartbeatConfig
  connectedChannels: ConnectedChannel[]
  onHeartbeatChange: HeartbeatChange
}

function HeartbeatScheduleFields({ heartbeat, connectedChannels, onHeartbeatChange }: HeartbeatScheduleFieldsProps) {
  return (
    <>
      {/* Interval & Hours */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Interval
            <InlineHelp id="settings.heartbeat.interval" size="sm" />
          </Label>
          <Select
            value={String(heartbeat.interval_minutes)}
            onValueChange={(v) => onHeartbeatChange('interval_minutes', parseInt(v))}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="15">Every 15 minutes</SelectItem>
              <SelectItem value="30">Every 30 minutes</SelectItem>
              <SelectItem value="60">Every hour</SelectItem>
              <SelectItem value="120">Every 2 hours</SelectItem>
              <SelectItem value="240">Every 4 hours</SelectItem>
              <SelectItem value="480">Every 8 hours</SelectItem>
              <SelectItem value="1440">Daily</SelectItem>
              <SelectItem value="10080">Weekly</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Active From</Label>
          <Input
            type="time"
            value={heartbeat.active_hours_start}
            onChange={(e) => onHeartbeatChange('active_hours_start', e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label>Active Until</Label>
          <Input
            type="time"
            value={heartbeat.active_hours_end}
            onChange={(e) => onHeartbeatChange('active_hours_end', e.target.value)}
          />
        </div>
      </div>

      {/* Timezone & Notification */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Timezone</Label>
          <Input
            value={heartbeat.timezone}
            onChange={(e) => onHeartbeatChange('timezone', e.target.value)}
            placeholder="America/New_York"
          />
          <p className="text-xs text-muted-foreground">IANA timezone (auto-detected from browser)</p>
        </div>

        <div className="space-y-2">
          <Label className="flex items-center gap-1">
            <MessageSquare className="h-3 w-3" />
            Notification Channel
          </Label>
          <Select
            value={heartbeat.notification_channel}
            onValueChange={(v) => onHeartbeatChange('notification_channel', v)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="in_app">In-App Notification</SelectItem>
              {connectedChannels.map((ch) => (
                <SelectItem key={ch.key} value={ch.key}>
                  {ch.platform.charAt(0).toUpperCase() + ch.platform.slice(1)}
                </SelectItem>
              ))}
              <SelectItem value="webhook">Webhook URL</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Checklist */}
      <div className="space-y-2">
        <Label>Heartbeat Checklist</Label>
        <Textarea
          value={heartbeat.checklist}
          onChange={(e) => onHeartbeatChange('checklist', e.target.value)}
          placeholder="- Check agent health status&#10;- Review pending tasks&#10;- Summarize today's activity"
          rows={5}
          className="font-mono text-sm"
        />
        <p className="text-xs text-muted-foreground">
          What should the orchestrator check each heartbeat? One item per line.
        </p>
      </div>
    </>
  )
}

interface HeartbeatRunPanelProps {
  heartbeatRunning: boolean
  lastHeartbeatResult: any
  onRunNow: () => void
}

function HeartbeatRunPanel({ heartbeatRunning, lastHeartbeatResult, onRunNow }: HeartbeatRunPanelProps) {
  return (
    <>
      {/* Run Now & Last Result */}
      <div className="flex gap-2 pt-4 border-t border-border/30">
        <Button
          variant="outline"
          size="sm"
          onClick={onRunNow}
          disabled={heartbeatRunning}
        >
          {heartbeatRunning ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
          Run Now
        </Button>
      </div>

      {lastHeartbeatResult && (
        <div className="space-y-2 pt-4 border-t border-border/30">
          <Label className="text-xs">Last Heartbeat Result</Label>
          <div className="p-3 rounded-lg bg-secondary/30 text-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className={`h-2 w-2 rounded-full ${lastHeartbeatResult.status === 'success' ? 'bg-success' : 'bg-destructive'}`} />
              <span className="font-medium text-xs">{lastHeartbeatResult.status}</span>
              <span className="text-xs text-muted-foreground">
                {lastHeartbeatResult.created_at ? new Date(lastHeartbeatResult.created_at).toLocaleString() : ''}
              </span>
              {lastHeartbeatResult.tokens_used > 0 && (
                <Badge variant="outline" className="text-xs ml-auto">{lastHeartbeatResult.tokens_used} tokens</Badge>
              )}
            </div>
            <p className="text-xs whitespace-pre-wrap">
              {lastHeartbeatResult.findings?.find((f: any) => f.check === 'llm_analysis')?.detail
                || lastHeartbeatResult.findings?.[0]?.detail
                || 'No details available'}
            </p>
          </div>
        </div>
      )}
    </>
  )
}
