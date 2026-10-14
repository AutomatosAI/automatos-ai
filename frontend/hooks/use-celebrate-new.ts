'use client'

import { useEffect, useRef, useState } from 'react'

import type { Deliverable } from '@/hooks/use-deliverables-api'
import { freshDeliverables, markSeen, readSeen, type SeenRecord, writeSeen } from '@/lib/play/seen'

/**
 * PRD-260 §4 — Auto celebrates once when Play's shelf shows a Deliverable this
 * browser hasn't seen (lib/play/seen.ts decides what counts as new), ported
 * from Automatos Play's hooks/useCelebrateNew.ts. Returns the newest one that
 * set it off, for the caption ("Your report is ready").
 */
export function useCelebrateNew(items: readonly Deliverable[] | null, celebrate: () => void): Deliverable | null {
  // undefined until the first read: localStorage is browser-only.
  const seen = useRef<SeenRecord | null | undefined>(undefined)
  const [latest, setLatest] = useState<Deliverable | null>(null)

  useEffect(() => {
    if (!items || items.length === 0) return
    const before = seen.current === undefined ? readSeen() : seen.current
    const fresh = freshDeliverables(before, items)
    const after = markSeen(before, items)
    seen.current = after
    if (after !== before) writeSeen(after)
    const [newest] = fresh
    if (newest) {
      setLatest(newest)
      celebrate()
    }
  }, [items, celebrate])

  return latest
}
