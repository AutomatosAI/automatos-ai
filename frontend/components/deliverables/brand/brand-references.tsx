'use client'

/**
 * PRD-251B US-B302 — the brand kit's style references: images the brand likes, or wants
 * to avoid, each with a note on why. Up to the kit's limit, PNG, JPEG or WebP (the server
 * reads the type from the file and checks its size). A new one is liked; its stance and
 * note change in place, and it can be removed. Every change has Auto read them again.
 */
import { useRef, useState, type ChangeEvent } from 'react'
import { ImagePlus, Loader2, Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import type { BrandReference, BrandReferenceStance } from '@/lib/brand-style-types'
import { useAuthedImage } from '@/hooks/use-authed-image'
import type { useBrandStyle } from '@/hooks/use-brand-style'

export const REFERENCE_TYPES = 'image/png,image/jpeg,image/webp'
const STANCES: ReadonlyArray<{ value: BrandReferenceStance; label: string }> = [
  { value: 'like', label: 'Like' },
  { value: 'avoid', label: 'Avoid' },
]

type BrandStyle = ReturnType<typeof useBrandStyle>

interface ReferenceCardProps {
  reference: BrandReference
  index: number
  canEdit: boolean
  style: BrandStyle
}

function ReferenceCard({ reference, index, canEdit, style }: ReferenceCardProps) {
  const image = useAuthedImage(reference.url)
  const [note, setNote] = useState(reference.note)
  const name = `Reference ${index + 1}`
  const saveNote = () => {
    if (note.trim() !== reference.note) style.update.mutate({ id: reference.id, changes: { note: note.trim() } })
  }
  return (
    <li className="flex flex-col gap-2 rounded-lg border bg-background/60 p-2" aria-label={name}>
      <div className="relative aspect-[4/3] overflow-hidden rounded-md bg-muted">
        {image && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={image} alt={reference.note || name} className="h-full w-full object-cover" />
        )}
        <span className={cn('absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium', reference.stance === 'like' ? 'bg-emerald-600 text-white' : 'bg-rose-600 text-white')}>
          {reference.stance === 'like' ? 'Liked' : 'Avoid'}
        </span>
      </div>
      {canEdit ? (
        <>
          <div role="group" aria-label={`${name}: like or avoid`} className="flex gap-1">
            {STANCES.map((stance) => (
              <Button
                key={stance.value} type="button" size="sm" variant={reference.stance === stance.value ? 'default' : 'outline'}
                aria-pressed={reference.stance === stance.value} className="h-7 flex-1 text-xs"
                onClick={() => reference.stance !== stance.value && style.update.mutate({ id: reference.id, changes: { stance: stance.value } })}
              >
                {stance.label}
              </Button>
            ))}
          </div>
          <Input aria-label={`${name} note`} value={note} maxLength={300} placeholder="Why: the light, the framing…" onChange={(e) => setNote(e.target.value)} onBlur={saveNote} className="h-8 text-xs" />
          <Button type="button" size="sm" variant="ghost" className="h-7 gap-1 text-xs text-destructive" aria-label={`Remove ${name}`} onClick={() => style.remove.mutate(reference.id)}>
            <Trash2 className="h-3.5 w-3.5" aria-hidden /> Remove
          </Button>
        </>
      ) : (
        reference.note && <p className="text-xs text-muted-foreground">{reference.note}</p>
      )}
    </li>
  )
}

interface BrandReferencesProps {
  style: BrandStyle
  canEdit: boolean
}

export function BrandReferences({ style, canEdit }: BrandReferencesProps) {
  const input = useRef<HTMLInputElement | null>(null)
  const data = style.query.data
  const references = data?.references ?? []
  const limit = data?.limits.references ?? 0
  const full = !!data && references.length >= limit
  const pick = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (file) style.upload.mutate({ file, note: '', stance: 'like' })
  }
  return (
    <section aria-label="Style references" className="flex flex-col gap-3 rounded-xl border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Style references</h3>
          <p className="text-xs text-muted-foreground">
            Images you like, and ones to avoid, with a note on why. {data ? `${references.length} of ${limit}.` : ''}
          </p>
        </div>
        {canEdit && (
          <>
            <input ref={input} type="file" accept={REFERENCE_TYPES} className="hidden" aria-label="Style reference file" onChange={pick} />
            <Button type="button" size="sm" variant="outline" className="gap-1.5" disabled={full || style.upload.isLoading} onClick={() => input.current?.click()}>
              {style.upload.isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> : <ImagePlus className="h-3.5 w-3.5" aria-hidden />}
              Add an image
            </Button>
          </>
        )}
      </div>
      {style.query.isLoading && <p className="text-sm text-muted-foreground">Loading the references…</p>}
      {style.query.isError && <p className="text-sm text-destructive">The references could not be loaded.</p>}
      {data && references.length === 0 && (
        <p className="text-sm text-muted-foreground">No references yet. Add a few images in the style you want, and one or two you never want to look like.</p>
      )}
      {references.length > 0 && (
        <ul className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {references.map((reference, index) => (
            <ReferenceCard key={reference.id} reference={reference} index={index} canEdit={canEdit} style={style} />
          ))}
        </ul>
      )}
      {full && canEdit && <p className="text-xs text-muted-foreground">The kit holds {limit} references at most: remove one to add another.</p>}
    </section>
  )
}
