'use client'

import * as React from 'react'
import { useFormContext } from 'react-hook-form'
import { Play, Clock, Webhook, Copy, Check } from 'lucide-react'
import { Label } from '@/components/ui/label'
import { InlineHelp } from '@/components/ui/help-tooltip'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import type { PlaybookFormValues } from './create-playbook-modal'

const SCHEDULE_TYPES = [
  { value: 'manual', label: 'Manual', icon: Play, description: 'Run on demand with a button click' },
  { value: 'cron', label: 'Scheduled', icon: Clock, description: 'Run on a cron schedule' },
  { value: 'trigger', label: 'Triggered', icon: Webhook, description: 'Run via webhook or external event' },
] as const

const CRON_QUICK_PICKS = [
  { label: 'Every hour', value: '0 * * * *' },
  { label: 'Daily at 9am', value: '0 9 * * *' },
  { label: 'Weekdays at 9am', value: '0 9 * * 1-5' },
  { label: 'Weekly on Monday', value: '0 9 * * 1' },
  { label: 'Custom', value: '' },
]

function getNextCronRuns(expression: string, count: number): string[] {
  // Simple cron parser for preview — handles common patterns
  const parts = expression.trim().split(/\s+/)
  if (parts.length !== 5) return []

  const now = new Date()
  const runs: string[] = []

  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts

  // Generate next N dates by iterating forward
  const candidate = new Date(now)
  candidate.setSeconds(0)
  candidate.setMilliseconds(0)
  candidate.setMinutes(candidate.getMinutes() + 1)

  let iterations = 0
  const maxIterations = 60 * 24 * 31 // check up to ~1 month of minutes

  while (runs.length < count && iterations < maxIterations) {
    const m = candidate.getMinutes()
    const h = candidate.getHours()
    const dom = candidate.getDate()
    const mon = candidate.getMonth() + 1
    const dow = candidate.getDay() === 0 ? 7 : candidate.getDay() // 1=Mon ... 7=Sun

    const matchMinute = minute === '*' || parseInt(minute) === m
    const matchHour = hour === '*' || parseInt(hour) === h
    const matchDom = dayOfMonth === '*' || parseInt(dayOfMonth) === dom
    const matchMonth = month === '*' || parseInt(month) === mon

    let matchDow = false
    if (dayOfWeek === '*') {
      matchDow = true
    } else if (dayOfWeek.includes('-')) {
      const [start, end] = dayOfWeek.split('-').map(Number)
      matchDow = dow >= start && dow <= end
    } else {
      matchDow = parseInt(dayOfWeek) === dow
    }

    if (matchMinute && matchHour && matchDom && matchMonth && matchDow) {
      runs.push(
        candidate.toLocaleString('en-US', {
          weekday: 'short',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        })
      )
    }

    candidate.setMinutes(candidate.getMinutes() + 1)
    iterations++
  }

  return runs
}

interface PlaybookScheduleConfigProps {
  webhookId?: string
}

export function PlaybookScheduleConfig({ webhookId }: PlaybookScheduleConfigProps) {
  const methods = useFormContext<PlaybookFormValues>()
  const scheduleConfig = methods.watch('schedule_config')
  const [copied, setCopied] = React.useState(false)
  const [cronQuick, setCronQuick] = React.useState('')

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  const updateSchedule = (field: keyof PlaybookFormValues['schedule_config'], value: unknown) => {
    methods.setValue('schedule_config', { ...scheduleConfig, [field]: value })
  }

  const webhookUrl = webhookId
    ? `${apiBaseUrl}/api/webhooks/recipe/${webhookId}`
    : null

  const handleCopyWebhook = () => {
    if (!webhookUrl) return
    navigator.clipboard.writeText(webhookUrl)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleQuickPick = (value: string) => {
    setCronQuick(value)
    if (value) {
      updateSchedule('cron_expression', value)
    }
  }

  const nextRuns = React.useMemo(() => {
    if (scheduleConfig.type !== 'cron' || !scheduleConfig.cron_expression) return []
    return getNextCronRuns(scheduleConfig.cron_expression, 5)
  }, [scheduleConfig.type, scheduleConfig.cron_expression])

  return (
    <div className="space-y-4">
      {/* Schedule Type Selection */}
      <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
        <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
          <span className="text-base">📅</span> Execution Type
        </h4>
        <div className="grid grid-cols-3 gap-3">
          {SCHEDULE_TYPES.map((st) => {
            const Icon = st.icon
            return (
              <button
                key={st.value}
                type="button"
                onClick={() => updateSchedule('type', st.value)}
                className={`p-4 rounded-xl border text-left text-sm transition-all duration-200 ${
                  scheduleConfig.type === st.value
                    ? 'border-primary bg-primary/10 text-foreground'
                    : 'border-border/30 bg-secondary/50 text-muted-foreground hover:border-border/60'
                }`}
              >
                <div className="flex items-center gap-2 font-medium">
                  <Icon className="w-4 h-4" />
                  {st.label}
                </div>
                <div className="text-xs mt-1.5 opacity-70">{st.description}</div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Manual Mode */}
      {scheduleConfig.type === 'manual' && (
        <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
          <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <span className="text-base">▶</span> Manual Execution
          </h4>
          <p className="text-sm text-muted-foreground">
            This recipe will be executed manually. Use the Execute button on the playbook detail page or call the API directly.
          </p>
          <div className="p-4 rounded-xl bg-secondary/30 border border-border/20">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
              <Play className="w-3.5 h-3.5" />
              <span>Test Run</span>
            </div>
            <p className="text-xs text-muted-foreground/70">
              After saving, you can test this playbook from the playbook detail view using the Execute button.
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled
              className="mt-3 text-xs opacity-60"
            >
              <Play className="w-3 h-3 mr-1.5" />
              Test Run (Save first)
            </Button>
          </div>
        </div>
      )}

      {/* Scheduled (Cron) Mode */}
      {scheduleConfig.type === 'cron' && (
        <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
          <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <span className="text-base">🕐</span> Schedule Configuration
          </h4>

          {/* Quick Picks */}
          <div>
            <Label className="text-xs text-muted-foreground">Quick Picks</Label>
            <select
              value={cronQuick}
              onChange={(e) => handleQuickPick(e.target.value)}
              className="w-full mt-1 bg-secondary/50 rounded-xl border border-border/30 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 appearance-none cursor-pointer"
            >
              <option value="">Select a schedule...</option>
              {CRON_QUICK_PICKS.map((pick) => (
                <option key={pick.label} value={pick.value}>
                  {pick.label}{pick.value ? ` (${pick.value})` : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Cron Expression */}
          <div>
            <Label className="text-xs text-muted-foreground flex items-center gap-1">Cron Expression <InlineHelp id="recipes.schedule.cron_expression" size="sm" /></Label>
            <Input
              value={scheduleConfig.cron_expression || ''}
              onChange={(e) => {
                updateSchedule('cron_expression', e.target.value)
                setCronQuick('')
              }}
              placeholder="0 9 * * 1-5"
              className="mt-1 bg-secondary/50 rounded-xl font-mono"
            />
            <div className="flex items-center gap-4 text-xs text-muted-foreground/60 mt-1">
              <span>minute</span>
              <span>hour</span>
              <span>day(month)</span>
              <span>month</span>
              <span>day(week)</span>
            </div>
          </div>

          {/* Next 5 Runs Preview */}
          {nextRuns.length > 0 && (
            <div className="p-4 rounded-xl bg-secondary/30 border border-border/20">
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                <Clock className="w-3.5 h-3.5" />
                <span>Next 5 runs</span>
              </div>
              <ul className="space-y-1">
                {nextRuns.map((run, i) => (
                  <li key={i} className="text-xs text-muted-foreground/80 font-mono flex items-center gap-2">
                    <span className="w-4 h-4 flex items-center justify-center rounded-full bg-primary/20 text-primary text-[10px]">{i + 1}</span>
                    {run}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {scheduleConfig.cron_expression && nextRuns.length === 0 && (
            <div className="p-2.5 rounded-xl bg-[hsl(var(--destructive))]/10 border border-[hsl(var(--destructive))]/20 text-xs text-[hsl(var(--destructive))]">
              Invalid cron expression or no upcoming runs found
            </div>
          )}
        </div>
      )}

      {/* Triggered (Webhook) Mode */}
      {scheduleConfig.type === 'trigger' && (
        <div className="space-y-4">
          {/* Webhook URL */}
          <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <span className="text-base">🔗</span> Webhook URL
            </h4>

            {webhookUrl ? (
              <>
                <div className="flex items-center gap-2">
                  <div className="flex-1 relative">
                    <Input
                      readOnly
                      value={webhookUrl}
                      className="bg-secondary/50 rounded-xl font-mono text-xs pr-10"
                    />
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={handleCopyWebhook}
                    className={`shrink-0 rounded-xl transition-all duration-200 ${
                      copied
                        ? 'border-[hsl(var(--success))] text-[hsl(var(--success))]'
                        : 'border-primary/50 hover:border-primary hover:shadow-[0_0_12px_hsl(var(--primary)/0.3)]'
                    }`}
                  >
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground/60">
                  Configure this URL as a webhook in your external service (GitHub, Jira, Slack, etc.).
                  The request body will be passed as input data to the recipe.
                </p>
              </>
            ) : (
              <div className="p-4 rounded-xl bg-secondary/30 border border-border/20 text-center">
                <Webhook className="w-6 h-6 mx-auto mb-2 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground/70">
                  Save the recipe to generate a webhook URL
                </p>
                <p className="text-xs text-muted-foreground/50 mt-1">
                  A unique, persistent URL will be created when this playbook is saved.
                </p>
              </div>
            )}
          </div>

          {/* POST Example */}
          <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <span className="text-base">⚡</span> Usage
            </h4>
            <div className="p-4 rounded-xl bg-secondary/30 border border-border/20 space-y-2">
              <p className="text-xs text-muted-foreground">
                Send a POST request to trigger this playbook. The JSON body is passed as input data.
              </p>
              <div className="font-mono text-xs text-muted-foreground/80 bg-secondary/50 rounded-lg p-3">
                <span className="text-primary">POST</span> {webhookUrl || `{webhook_url}`}
                <br />
                <span className="text-muted-foreground/50">Content-Type:</span> application/json
                <br />
                <span className="text-muted-foreground/50">Body:</span> {'{ "your_input": "data" }'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
