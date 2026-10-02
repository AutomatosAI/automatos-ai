'use client'

/**
 * PRD-251 S2.2c (US-209) — the options a channel's platform needs: YouTube's privacy
 * and category (the editor's channel line shows them once YouTube is ticked, PRD-251B
 * US-B109). LinkedIn posts as the member whose account is connected; TikTok's privacy
 * and AI label are chosen when it publishes (Wave 3).
 */
import { Label } from '@/components/ui/label'
import type { SocialPostTargetOptions } from '@/lib/api-client'

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

/** YouTube's privacy and category, the options its upload needs (US-209; the editor's channel row, PRD-251B US-B109). */
export function YoutubeOptions({ options, onChange }: { options: SocialPostTargetOptions; onChange: (o: SocialPostTargetOptions) => void }) {
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
