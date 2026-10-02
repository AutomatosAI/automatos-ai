'use client'

/**
 * PRD-251B US-B303 — What Auto takes from the references: the palette, the mood, how the
 * liked images are composed and what the avoided ones have. Read again by itself whenever
 * the references change, or on "Read the references again". It goes into the composer's,
 * a plan's research and Template Studio's image prompts; whether the liked images
 * themselves go to AI tools that take a reference image is the switch below it.
 */
import { Loader2, RefreshCw } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { BrandStyleProfile } from '@/lib/brand-style-types'
import type { useBrandStyle } from '@/hooks/use-brand-style'

type BrandStyle = ReturnType<typeof useBrandStyle>

function ProfileBody({ profile }: { profile: BrandStyleProfile }) {
  return (
    <div className="flex flex-col gap-3 text-sm">
      {profile.palette.length > 0 && (
        <ul aria-label="Palette" className="flex flex-wrap gap-2">
          {profile.palette.map((colour) => (
            <li key={colour} className="flex items-center gap-1.5 font-mono text-xs">
              <span className="h-5 w-5 rounded border" style={{ background: colour }} aria-hidden />
              {colour}
            </li>
          ))}
        </ul>
      )}
      {profile.mood.length > 0 && (
        <ul aria-label="Mood" className="flex flex-wrap gap-1.5">
          {profile.mood.map((word) => (
            <li key={word} className="rounded-full border px-2 py-0.5 text-xs">{word}</li>
          ))}
        </ul>
      )}
      {profile.composition && <p><span className="font-medium">Composition:</span> {profile.composition}</p>}
      {profile.avoid && <p><span className="font-medium">Avoid:</span> {profile.avoid}</p>}
    </div>
  )
}

interface BrandStyleProfileCardProps {
  style: BrandStyle
  canEdit: boolean
}

export function BrandStyleProfileCard({ style, canEdit }: BrandStyleProfileCardProps) {
  const data = style.query.data
  const hasReferences = (data?.references.length ?? 0) > 0
  const profile = data?.profile ?? null
  return (
    <aside aria-label="What Auto takes from these" className="flex flex-col gap-3 rounded-xl border bg-card p-4">
      <h3 className="text-sm font-semibold text-foreground">What Auto takes from these</h3>
      {style.reading && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Auto is reading the references…
        </p>
      )}
      {!hasReferences && <p className="text-sm text-muted-foreground">Add references: Auto reads their palette, mood and composition.</p>}
      {hasReferences && !profile && !style.reading && <p className="text-sm text-muted-foreground">Auto has not read these yet.</p>}
      {profile && <ProfileBody profile={profile} />}
      {canEdit && hasReferences && (
        <Button type="button" size="sm" variant="outline" className="w-fit gap-1.5" disabled={style.readAgain.isLoading} onClick={() => style.readAgain.mutate(undefined)}>
          {style.readAgain.isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> : <RefreshCw className="h-3.5 w-3.5" aria-hidden />}
          Read the references again
        </Button>
      )}
      {data && (
        <label className="flex items-start gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox" className="mt-0.5" checked={data.send_liked} disabled={!canEdit || style.sendLiked.isLoading}
            onChange={(e) => style.sendLiked.mutate(e.target.checked)}
          />
          Send liked images to AI tools whose action takes a reference image
        </label>
      )}
    </aside>
  )
}
