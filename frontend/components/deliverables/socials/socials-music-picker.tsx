'use client'

/**
 * PRD-251B US-B109 — the music a video post plays (Editor.dc.html's Music): the template's own
 * track, another track of the renderer's music library (by its style, "Library · deep
 * house"), or no music. Saved on the post and used by its next render: a render setting, so
 * it never voids an approval. A track that asks for credit gets its line in the copy.
 */
import type { SocialPost } from '@/lib/api-client'
import type { SocialMusicTrack, SocialPostMusic } from '@/lib/socials-plan-types'
import { useUpdateSocialPost } from '@/hooks/use-socials-api'
import { useSocialMusic } from '@/hooks/use-socials-plans'

const SELECT_CLASS = 'h-9 w-full rounded-md border border-input bg-background px-2 text-sm'
export const TEMPLATE_TRACK = ''
export const NO_MUSIC = '__none__'
export const NO_LIBRARY = 'The music library is not reachable: the template plays its own track.'

/** "Library · Deep house" (the track's style), else its title. */
export function trackLabel(track: Pick<SocialMusicTrack, 'style' | 'title'>): string {
  const name = track.style ? track.style.charAt(0).toUpperCase() + track.style.slice(1) : track.title
  return `Library · ${name}`
}

export function musicValue(music: SocialPostMusic | undefined): string {
  if (!music) return TEMPLATE_TRACK
  return music.track === null ? NO_MUSIC : music.track
}

export function musicFrom(value: string): SocialPostMusic {
  if (value === TEMPLATE_TRACK) return null
  return { track: value === NO_MUSIC ? null : value }
}

export function SocialsMusicPicker({ post, editable }: { post: SocialPost; editable: boolean }) {
  const { data } = useSocialMusic()
  const update = useUpdateSocialPost()
  const tracks = data?.tracks ?? []
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={`music-${post.id}`} className="text-sm font-medium text-foreground">Music</label>
      <select id={`music-${post.id}`} className={SELECT_CLASS} value={musicValue(post.music)} disabled={!editable || update.isLoading}
        onChange={(e) => update.mutate({ postId: post.id, changes: { music: musicFrom(e.target.value) } })}>
        <option value={TEMPLATE_TRACK}>The template&apos;s own track</option>
        {tracks.map((track) => <option key={track.id} value={track.id}>{trackLabel(track)}</option>)}
        <option value={NO_MUSIC}>No music</option>
      </select>
      {data && !data.available && <span className="text-[12.5px] text-muted-foreground">{NO_LIBRARY}</span>}
    </div>
  )
}
