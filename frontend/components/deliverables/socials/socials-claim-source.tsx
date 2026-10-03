'use client'

/**
 * PRD-251 S2.2b (US-208) — a claim's source (D7): the source it is bound to, or
 * the red Unsourced chip and a search of this workspace's candidates
 * (GET /api/socials/sources) to bind one.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SocialClaimSource } from '@/lib/api-client'
import { useSocialSourceSearch } from '@/hooks/use-socials-composer'
import { UnsourcedChip } from './socials-post-evidence'

interface ClaimSourceProps {
  name: string
  source: SocialClaimSource | null
  onChange: (source: SocialClaimSource | null) => void
  /** PRD-251B US-B109: the editor's "Add a source" opens this claim's search. */
  initiallyPicking?: boolean
}

export function ClaimSource({ name, source, onChange, initiallyPicking = false }: ClaimSourceProps) {
  const [picking, setPicking] = useState(initiallyPicking)
  const [q, setQ] = useState('')
  const search = useSocialSourceSearch(q.trim(), picking)
  const candidates = search.data?.candidates ?? []

  const pick = (candidate: SocialClaimSource) => {
    onChange({ kind: candidate.kind, ref: candidate.ref, as_of: candidate.as_of ?? null })
    setPicking(false)
    setQ('')
  }

  return (
    <div className="space-y-1.5" id={`socials-claim-source-${name}`} data-testid={`socials-claim-source-${name}`}>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {source ? (
          <span className="text-muted-foreground">
            Source: {source.kind} {source.ref}
            {source.as_of ? `, as of ${source.as_of}` : ''}
          </span>
        ) : (
          <UnsourcedChip />
        )}
        <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={() => setPicking(!picking)}>
          {source ? 'Change source' : 'Pick a source'}
        </Button>
      </div>
      {picking && (
        <div className="space-y-1 rounded-lg border border-border/60 p-2">
          <Input
            aria-label={`Search sources for ${name}`}
            value={q}
            onChange={(event) => setQ(event.target.value)}
            placeholder="A report, a Deliverable, a metric or a link"
          />
          {candidates.length === 0 ? (
            <p className="text-xs text-muted-foreground">{search.isLoading ? 'Searching…' : 'Nothing found.'}</p>
          ) : (
            <ul className="max-h-40 space-y-1 overflow-y-auto">
              {candidates.map((candidate) => (
                <li key={`${candidate.kind}-${candidate.ref}`}>
                  <button
                    type="button"
                    className="w-full rounded px-2 py-1 text-left text-xs hover:bg-muted"
                    onClick={() => pick(candidate)}
                  >
                    <span className="font-medium text-foreground">{candidate.title}</span>
                    <span className="text-muted-foreground"> · {candidate.kind}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
