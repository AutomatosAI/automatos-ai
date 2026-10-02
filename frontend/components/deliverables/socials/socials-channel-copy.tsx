'use client'

/**
 * PRD-251 S2.2c (US-209) — one channel's copy with its live count against the
 * channel's limits (GET /api/socials/channels copy_limits): red over the limit,
 * which blocks submit. Instagram counts its hashtags too. The editor's preview shows it
 * under each channel's tab as "Copy for <channel>" (PRD-251B US-B110).
 */
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { SocialCopyLimits } from '@/lib/api-client'
import { copyCount } from './socials-composer-model'

interface ChannelCopyFieldProps {
  toolkit: string
  label: string
  value: string
  limits: SocialCopyLimits | null | undefined
  onChange: (text: string) => void
}

export function ChannelCopyField({ toolkit, label, value, limits, onChange }: ChannelCopyFieldProps) {
  const id = `socials-composer-copy-${toolkit}`
  const count = copyCount(value, limits)
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>Copy for {label}</Label>
      <Textarea id={id} value={value} rows={3} aria-invalid={count.over} onChange={(event) => onChange(event.target.value)} />
      <p
        data-testid={`socials-copy-count-${toolkit}`}
        className={cn('text-right text-xs', count.over ? 'font-medium text-destructive' : 'text-muted-foreground')}
      >
        {count.limit !== null ? `${count.count} / ${count.limit}` : `${count.count} characters`}
        {count.hashtagLimit !== null && ` · ${count.hashtags} / ${count.hashtagLimit} hashtags`}
      </p>
    </div>
  )
}
