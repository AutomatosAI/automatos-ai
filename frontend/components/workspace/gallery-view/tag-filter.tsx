/**
 * TagFilter (7 Oct)
 * =================
 *
 * The Deliverables filter bar's tag box: the owner types a tag and the list keeps the
 * Deliverables carrying it (GET /api/deliverables?tag=). Debounced like the search box;
 * the tag is read as the platform stores tags (trimmed, one space, lowercase), and an
 * empty box clears the filter.
 */

'use client'

import { useEffect, useState } from 'react'
import { Tag } from 'lucide-react'

import { Input } from '@/components/ui/input'

import { useDebouncedValue } from './use-debounced-value'

export const TAG_FILTER_DEBOUNCE_MS = 300
// The longest tag the platform keeps (services/deliverable_tags.py MAX_TAG_CHARS).
export const MAX_TAG_CHARS = 40

/** The tag as the platform stores it, or null for an empty box. */
export function cleanTag(raw: string): string | null {
  const tag = raw.trim().replace(/\s+/g, ' ').toLowerCase()
  return tag || null
}

export interface TagFilterProps {
  value: string | null | undefined
  onChange: (tag: string | null) => void
}

export function TagFilter({ value, onChange }: TagFilterProps) {
  const [input, setInput] = useState(value ?? '')
  const debounced = useDebouncedValue(input, TAG_FILTER_DEBOUNCE_MS)
  const current = value ?? null

  // A change from outside (Clear) resets the box; the owner's own typing is left as typed.
  useEffect(() => {
    setInput((typed) => (cleanTag(typed) === current ? typed : current ?? ''))
  }, [current])

  useEffect(() => {
    const tag = cleanTag(debounced)
    if (tag !== current) onChange(tag)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debounced])

  return (
    <div className="relative w-[170px]">
      <Tag className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Tag"
        className="pl-9"
        aria-label="Filter by tag"
        maxLength={MAX_TAG_CHARS}
      />
    </div>
  )
}
