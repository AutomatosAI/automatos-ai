'use client'

/**
 * PRD-251B US-B304 — the Brand kit tab's AI tools: each media toolkit the capability
 * registry can offer (connected, to connect in Composio, unavailable and why), Templates
 * and Kokoro built in and free; the default per media type, from what is offered now; the
 * monthly media cap and the per-post cap, with this month's spend. Paid tools are connected
 * in Composio only (D15). Owners and admins change them; the server checks every choice.
 * F252: a dropdown with one choice says why, and what to connect for more.
 */
import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { SocialMediaToolkitRow, SocialMediaToolsResponse, SocialMediaType } from '@/lib/brand-style-types'
import { useSocialMediaTools, useUpdateSocialMediaTools } from '@/hooks/use-media-tools'
import { useSocialsOn } from '@/hooks/use-socials-api'
import { ConnectToolkit } from '../socials/connect-toolkit'
import { oneChoiceHint } from './brand-ai-tools-model'

export const MEDIA_TYPE_LABELS: Record<SocialMediaType, string> = {
  images: 'Images',
  ai_images: 'AI images',
  footage: 'Footage',
  voice: 'Voice',
}
const MEDIA_TYPES = Object.keys(MEDIA_TYPE_LABELS) as SocialMediaType[]
const SELECT_CLASS = 'h-9 w-full rounded-md border border-input bg-background px-2 text-sm'
export const SOCIALS_OFF_NOTE = 'AI tools are part of Socials: turn Socials on in the Socials tab to choose them.'

function usd(value: number): string {
  return `$${value.toFixed(2)}`
}

function ToolkitState({ row, canEdit }: { row: SocialMediaToolkitRow; canEdit: boolean }) {
  if (row.status === 'connect') {
    return canEdit ? <ConnectToolkit toolkit={row.toolkit} label={row.label} purpose={`${row.label} here`} /> : <span>Not connected</span>
  }
  if (row.status === 'unavailable') return <span>Unavailable: {row.reason}</span>
  return <span>{row.status === 'builtin' ? 'Built in · free' : 'Connected'}</span>
}

function ToolkitRows({ rows, canEdit }: { rows: SocialMediaToolkitRow[]; canEdit: boolean }) {
  return (
    <ul aria-label="Media toolkits" className="divide-y rounded-lg border">
      {rows.map((row) => (
        <li key={`${row.kind}-${row.toolkit}`} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-sm">
          <span className="font-medium text-foreground">{row.label} <span className="text-xs font-normal text-muted-foreground">· {row.kind}</span></span>
          <span className="text-xs text-muted-foreground"><ToolkitState row={row} canEdit={canEdit} /></span>
        </li>
      ))}
    </ul>
  )
}

interface ToolsFormProps {
  tools: SocialMediaToolsResponse
  canEdit: boolean
}

function ToolsForm({ tools, canEdit }: ToolsFormProps) {
  const update = useUpdateSocialMediaTools()
  const [defaults, setDefaults] = useState(tools.defaults)
  const [monthly, setMonthly] = useState(String(tools.caps.monthly_usd))
  const [perPost, setPerPost] = useState(String(tools.caps.per_post_usd))
  useEffect(() => {
    setDefaults(tools.defaults)
    setMonthly(String(tools.caps.monthly_usd))
    setPerPost(String(tools.caps.per_post_usd))
  }, [tools])
  // Only the defaults changed here: one stored before a toolkit was disconnected is left as it is.
  const changed = Object.fromEntries(MEDIA_TYPES.filter((type) => defaults[type] !== tools.defaults[type]).map((type) => [type, defaults[type]]))
  const save = () => update.mutate({
    ...(Object.keys(changed).length ? { defaults: changed } : {}), monthly_cap_usd: Number(monthly), per_post_cap_usd: Number(perPost),
  })
  const capsValid = [monthly, perPost].every((value) => value.trim() !== '' && Number.isFinite(Number(value)) && Number(value) >= 0)
  return (
    <fieldset disabled={!canEdit} className="grid gap-3 sm:grid-cols-2">
      {MEDIA_TYPES.map((type) => {
        const hint = oneChoiceHint(type, tools)
        return (
          <div key={type}>
            <Label htmlFor={`media-default-${type}`} className="text-xs">{MEDIA_TYPE_LABELS[type]}</Label>
            <select
              id={`media-default-${type}`} className={SELECT_CLASS} value={defaults[type]} aria-describedby={hint ? `media-default-${type}-hint` : undefined}
              onChange={(e) => setDefaults({ ...defaults, [type]: e.target.value })}
            >
              {!(tools.offered[type] ?? []).some((choice) => choice.value === defaults[type]) && (
                <option value={defaults[type]}>{defaults[type]} (not available now)</option>
              )}
              {(tools.offered[type] ?? []).map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
            </select>
            {hint && <p id={`media-default-${type}-hint`} className="mt-1 text-xs text-muted-foreground">{hint}</p>}
          </div>
        )
      })}
      <div>
        <Label htmlFor="media-cap-monthly" className="text-xs">Monthly media cap (USD)</Label>
        <Input id="media-cap-monthly" type="number" min={0} step="0.01" value={monthly} onChange={(e) => setMonthly(e.target.value)} />
      </div>
      <div>
        <Label htmlFor="media-cap-post" className="text-xs">Per-post media cap (USD)</Label>
        <Input id="media-cap-post" type="number" min={0} step="0.01" value={perPost} onChange={(e) => setPerPost(e.target.value)} />
      </div>
      <p className="text-xs text-muted-foreground sm:col-span-2">
        Spent this month: {usd(tools.spend.month_usd)} of {usd(tools.caps.monthly_usd)}, until {new Date(tools.spend.period_end).toLocaleDateString()}.
        {tools.caps.problem ? ` ${tools.caps.problem}` : ''}
      </p>
      {canEdit && (
        <div className="flex justify-end sm:col-span-2">
          <Button type="button" onClick={save} disabled={update.isLoading || !capsValid}>
            {update.isLoading ? 'Saving…' : 'Save AI tools'}
          </Button>
        </div>
      )}
    </fieldset>
  )
}

export function BrandAiTools({ canEdit }: { canEdit: boolean }) {
  const { socialsOn } = useSocialsOn()
  const tools = useSocialMediaTools()
  return (
    <section aria-label="AI tools" className="flex flex-col gap-3 rounded-xl border bg-card p-4">
      <div>
        <h3 className="text-sm font-semibold text-foreground">AI tools</h3>
        <p className="text-xs text-muted-foreground">
          Templates and Kokoro are free and built in. Paid tools use your own account, connected in Composio.
        </p>
      </div>
      {!socialsOn && <p className="text-sm text-muted-foreground">{SOCIALS_OFF_NOTE}</p>}
      {socialsOn && tools.isLoading && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Loading the AI tools…</p>
      )}
      {socialsOn && tools.isError && <p className="text-sm text-destructive">The AI tools could not be loaded.</p>}
      {tools.data && (
        <>
          <ToolkitRows rows={tools.data.toolkits} canEdit={canEdit} />
          <ToolsForm tools={tools.data} canEdit={canEdit} />
        </>
      )}
    </section>
  )
}
