'use client'

import { useEffect, useRef, useState } from 'react'

import type { Deliverable } from '@automatos/theme-sdk'

// PRD-261 W-A5 seam item: the scope is the host's (it fills it), and the SDK has no name for it yet
// (context.useWorkspace carries no user). lib/__tests__/prd261-port-deliverables.test.ts allows
// exactly this import in exactly this file, and fails when the allowance outlives the import.
import { useLookScope } from '@/components/theme-host/look-scope'

import { freshDeliverables, markSeen, readSeen, type SeenRecord, writeSeen } from '../lib/play/seen'

/** The record as last read or written, and the scope it belongs to. */
interface ScopedSeen {
  readonly scope: string
  readonly record: SeenRecord | null
}

/**
 * PRD-260 §4 — Auto celebrates once when Play's shelf shows a Deliverable this
 * account hasn't seen in this browser (lib/play/seen.ts decides what counts as
 * new), ported from Automatos Play's hooks/useCelebrateNew.ts. The record is
 * the signed-in user's in this workspace (F5, useLookScope): nothing is read
 * or remembered until they are known, and another account starts its own.
 * Returns the newest one that set it off, for the caption ("Your report is ready").
 */
export function useCelebrateNew(items: readonly Deliverable[] | null, celebrate: () => void): Deliverable | null {
  const scope = useLookScope()
  // undefined until the first read: localStorage is browser-only.
  const seen = useRef<ScopedSeen | undefined>(undefined)
  const [latest, setLatest] = useState<Deliverable | null>(null)

  useEffect(() => {
    if (!items || items.length === 0 || scope === null) return
    const cached = seen.current
    const before = cached !== undefined && cached.scope === scope ? cached.record : readSeen(scope)
    const fresh = freshDeliverables(before, items)
    const after = markSeen(before, items)
    seen.current = { scope, record: after }
    if (after !== before) writeSeen(scope, after)
    const [newest] = fresh
    if (newest) {
      setLatest(newest)
      celebrate()
    }
  }, [items, celebrate, scope])

  return latest
}
