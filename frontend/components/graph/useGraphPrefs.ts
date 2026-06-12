'use client'

/**
 * Per-user + per-workspace graph view preferences (PRD-165 S1, BINDING Q25).
 *
 * Q25: "chips collapsed by default, filter/color prefs persisted per
 * user+workspace." The store is keyed by workspace + surface and lives in
 * localStorage (per-user implicitly — it's that user's browser), so a user's
 * legend/colour/filter choices survive a reload and don't bleed across
 * workspaces or between the KG / codegraph / mission surfaces.
 *
 * The read/write core is pure and SSR-safe so it unit-tests without React or a
 * DOM (S1 acceptance: "vitest for pref store").
 */

import { useCallback, useEffect, useState } from 'react'

export type GraphColorMode = 'type' | 'community'

export interface GraphPrefs {
  /** Q25: legend (filter chips) starts collapsed. */
  legendCollapsed: boolean
  colorMode: GraphColorMode
  /** Node `file_type` values the user has hidden. Empty = show all. */
  hiddenTypes: string[]
  /** Edge `relation` values the user has hidden. Empty = show all. */
  hiddenRelations: string[]
}

export const DEFAULT_GRAPH_PREFS: GraphPrefs = {
  legendCollapsed: true,
  colorMode: 'community',
  hiddenTypes: [],
  hiddenRelations: [],
}

/** Distinct surfaces that each keep their own prefs. */
export type GraphSurface = 'knowledge' | 'codegraph' | 'mission' | 'field'

const KEY_PREFIX = 'automatos:graph-prefs'

function storageKey(workspaceId: string | null | undefined, surface: GraphSurface): string {
  return `${KEY_PREFIX}:${workspaceId || 'default'}:${surface}`
}

/** Pure, SSR-safe read. Unknown/typo'd persisted fields are dropped; missing
 *  fields fall back to defaults, so an older saved blob never breaks. */
export function readGraphPrefs(
  workspaceId: string | null | undefined,
  surface: GraphSurface,
): GraphPrefs {
  if (typeof window === 'undefined') return { ...DEFAULT_GRAPH_PREFS }
  try {
    const raw = window.localStorage.getItem(storageKey(workspaceId, surface))
    if (!raw) return { ...DEFAULT_GRAPH_PREFS }
    const parsed = JSON.parse(raw) as Partial<GraphPrefs>
    return {
      legendCollapsed:
        typeof parsed.legendCollapsed === 'boolean'
          ? parsed.legendCollapsed
          : DEFAULT_GRAPH_PREFS.legendCollapsed,
      colorMode: parsed.colorMode === 'type' ? 'type' : 'community',
      hiddenTypes: Array.isArray(parsed.hiddenTypes)
        ? parsed.hiddenTypes.filter((t): t is string => typeof t === 'string')
        : [],
      hiddenRelations: Array.isArray(parsed.hiddenRelations)
        ? parsed.hiddenRelations.filter((r): r is string => typeof r === 'string')
        : [],
    }
  } catch {
    return { ...DEFAULT_GRAPH_PREFS }
  }
}

/** Pure, SSR-safe write. Swallows quota/serialisation errors — a failed pref
 *  save must never take down the graph. */
export function writeGraphPrefs(
  workspaceId: string | null | undefined,
  surface: GraphSurface,
  prefs: GraphPrefs,
): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(storageKey(workspaceId, surface), JSON.stringify(prefs))
  } catch {
    /* ignore — best-effort persistence */
  }
}

/**
 * React binding over the pure store. Returns the current prefs and a partial
 * updater that merges, persists, and re-renders. Re-reads when the workspace
 * or surface changes so switching workspace loads that workspace's prefs.
 */
export function useGraphPrefs(
  workspaceId: string | null | undefined,
  surface: GraphSurface,
): [GraphPrefs, (patch: Partial<GraphPrefs>) => void] {
  const [prefs, setPrefs] = useState<GraphPrefs>(() => readGraphPrefs(workspaceId, surface))

  useEffect(() => {
    setPrefs(readGraphPrefs(workspaceId, surface))
  }, [workspaceId, surface])

  const update = useCallback(
    (patch: Partial<GraphPrefs>) => {
      setPrefs((prev) => {
        const next = { ...prev, ...patch }
        writeGraphPrefs(workspaceId, surface, next)
        return next
      })
    },
    [workspaceId, surface],
  )

  return [prefs, update]
}
