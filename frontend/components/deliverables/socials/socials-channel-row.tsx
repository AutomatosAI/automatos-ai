'use client'

/**
 * PRD-251 S2.2c (US-209) — one connected channel in the composer: whether the
 * post goes there, the post kind it goes as (only available kinds can be chosen;
 * an unavailable one shows why), the channel's setup note, and the options its
 * platform needs: YouTube's privacy and category. LinkedIn posts as the member
 * whose account is connected; TikTok's privacy and AI label are chosen when it
 * publishes (Wave 3).
 */
import { Label } from '@/components/ui/label'
import type { SocialChannel, SocialPostKind, SocialPostTargetOptions } from '@/lib/api-client'

// YouTube's privacy statuses, and the categories a post is most often filed under
// (YouTube Data API videoCategories: their ids).
export const YOUTUBE_PRIVACY = ['public', 'unlisted', 'private'] as const
export const YOUTUBE_CATEGORIES: ReadonlyArray<{ id: string; label: string }> = [
  { id: '22', label: 'People & Blogs' },
  { id: '28', label: 'Science & Technology' },
  { id: '27', label: 'Education' },
  { id: '24', label: 'Entertainment' },
  { id: '26', label: 'Howto & Style' },
  { id: '25', label: 'News & Politics' },
  { id: '17', label: 'Sports' },
  { id: '19', label: 'Travel & Events' },
]
export const YOUTUBE_TOOLKIT = 'youtube'
export const DEFAULT_YOUTUBE_OPTIONS: SocialPostTargetOptions = { privacy_status: 'private', category_id: '22' }

function YoutubeOptions({ options, onChange }: { options: SocialPostTargetOptions; onChange: (o: SocialPostTargetOptions) => void }) {
  const current = { ...DEFAULT_YOUTUBE_OPTIONS, ...options }
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <div className="space-y-1">
        <Label htmlFor="socials-youtube-privacy">YouTube privacy</Label>
        <select
          id="socials-youtube-privacy"
          className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
          value={String(current.privacy_status)}
          onChange={(e) => onChange({ ...current, privacy_status: e.target.value })}
        >
          {YOUTUBE_PRIVACY.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>
      <div className="space-y-1">
        <Label htmlFor="socials-youtube-category">YouTube category</Label>
        <select
          id="socials-youtube-category"
          className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
          value={String(current.category_id)}
          onChange={(e) => onChange({ ...current, category_id: e.target.value })}
        >
          {YOUTUBE_CATEGORIES.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
        </select>
      </div>
    </div>
  )
}

interface ChannelRowProps {
  channel: SocialChannel
  kind: SocialPostKind | null
  options: SocialPostTargetOptions
  onToggle: (on: boolean) => void
  onKind: (kind: SocialPostKind) => void
  onOptions: (options: SocialPostTargetOptions) => void
}

export function ChannelRow({ channel, kind, options, onToggle, onKind, onOptions }: ChannelRowProps) {
  const anyAvailable = channel.post_kinds.some((k) => k.available)
  return (
    <li className="space-y-2 rounded-lg border border-border/60 p-3" data-testid={`socials-channel-${channel.toolkit}`}>
      <label className="flex items-center gap-2 text-sm font-medium text-foreground">
        <input type="checkbox" checked={kind !== null} disabled={!anyAvailable} onChange={(e) => onToggle(e.target.checked)} />
        {/* The label says "(unverified channel)" itself until the channel has published (US-305). */}
        {channel.label}
      </label>
      {channel.setup_note && <p className="text-xs text-muted-foreground">{channel.setup_note}</p>}
      <div role="radiogroup" aria-label={`${channel.label} post kind`} className="flex flex-wrap gap-3">
        {channel.post_kinds.map((k) => (
          <label key={k.kind} className="flex items-center gap-1.5 text-sm">
            <input
              type="radio"
              name={`socials-kind-${channel.toolkit}`}
              value={k.kind}
              checked={kind === k.kind}
              disabled={!k.available}
              onChange={() => onKind(k.kind)}
            />
            <span className={k.available ? 'text-foreground' : 'text-muted-foreground'}>{k.kind}</span>
            {!k.available && k.reason && <span className="text-xs text-muted-foreground">— {k.reason}</span>}
          </label>
        ))}
      </div>
      {kind !== null && channel.toolkit === YOUTUBE_TOOLKIT && <YoutubeOptions options={options} onChange={onOptions} />}
    </li>
  )
}
