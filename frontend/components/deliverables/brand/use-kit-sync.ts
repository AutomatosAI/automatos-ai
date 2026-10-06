'use client'

/**
 * F372 (night 10c) — the Brand kit page against changes made elsewhere (Auto, the Brand
 * designer, the API). The page reads the kit's fields once; a Save from a page left open
 * used to write the old values back over the newer kit.
 *
 * * `loadedStamp`: the kit's `updated_at` as the page last read or saved it. Every save sends
 *   it (`if_updated_at`); the server answers 409 when the kit has moved since.
 * * `edited`: the page holds changes not saved yet.
 * * `conflict`: a save was refused because the kit moved.
 * * `expectOwnChange` / `adopt`: a change the page stores itself (an upload, a role reset)
 *   moves the stamp too; the next stamp read is adopted as the page's own, not taken as a
 *   change made elsewhere.
 */
import { useCallback, useRef, useState } from 'react'

import type { BrandKit } from '@/components/documents/blocks/types'

export function useKitSync() {
  const [loadedStamp, setLoadedStamp] = useState<string | undefined>(undefined)
  const [edited, setEdited] = useState(false)
  const [conflict, setConflict] = useState(false)
  const ownChangePending = useRef(false)

  /** The page now shows ``kit`` as the server holds it: loaded, reloaded or saved. */
  const synced = useCallback((kit: BrandKit) => {
    ownChangePending.current = false
    setLoadedStamp(kit.updated_at ?? '')
    setEdited(false)
    setConflict(false)
  }, [])
  const markEdited = useCallback(() => setEdited(true), [])
  const expectOwnChange = useCallback(() => { ownChangePending.current = true }, [])

  /** Take ``stamp`` as the page's own when a change it stored is pending; true when it did. */
  const adopt = useCallback((stamp: string) => {
    if (!ownChangePending.current) return false
    ownChangePending.current = false
    setLoadedStamp(stamp)
    return true
  }, [])

  return { loadedStamp, edited, conflict, setConflict, synced, markEdited, expectOwnChange, adopt }
}

export type KitSync = ReturnType<typeof useKitSync>
