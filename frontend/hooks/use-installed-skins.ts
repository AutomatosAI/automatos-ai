'use client'

/**
 * PRD-259 §4 — this workspace's installed skins (GET /api/skins/installed through apiClient).
 *
 * The style provider owns one copy: it loads at once when the page opens on a skin (to
 * repair the cached stylesheet or fall back), and otherwise when the Style dropdown opens.
 * `reload` resolves to the fresh list, and `skinsRef` holds it as soon as it arrives, so a
 * caller can choose a skin it has just imported in the same tick.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react'
import { apiClient } from '@/lib/api-client'
import { INSTALLED_SKINS_PATH, parseInstalledSkins, type InstalledSkin } from '@/lib/skins/installed'

export type InstalledSkinsStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface InstalledSkinsState {
  readonly status: InstalledSkinsStatus
  readonly skins: readonly InstalledSkin[]
}

export interface InstalledSkins extends InstalledSkinsState {
  readonly skinsRef: MutableRefObject<readonly InstalledSkin[]>
  /** The fresh list; on a failure the last good list (status 'error', logged). */
  readonly reload: () => Promise<readonly InstalledSkin[]>
}

const IDLE: InstalledSkinsState = { status: 'idle', skins: [] }

export function useInstalledSkins(loadNow: boolean): InstalledSkins {
  const [state, setState] = useState<InstalledSkinsState>(IDLE)
  const skinsRef = useRef<readonly InstalledSkin[]>([])

  const reload = useCallback(async (): Promise<readonly InstalledSkin[]> => {
    setState((previous) => ({ ...previous, status: 'loading' }))
    try {
      const skins = parseInstalledSkins(await apiClient.get(INSTALLED_SKINS_PATH))
      skinsRef.current = skins
      setState({ status: 'ready', skins })
      return skins
    } catch (error) {
      console.warn('[Skins] Could not load the installed skins.', error)
      setState((previous) => ({ ...previous, status: 'error' }))
      return skinsRef.current
    }
  }, [])

  const started = useRef(false)
  useEffect(() => {
    if (!loadNow || started.current) return
    started.current = true
    void reload()
  }, [loadNow, reload])

  return useMemo(() => ({ ...state, skinsRef, reload }), [state, reload])
}
