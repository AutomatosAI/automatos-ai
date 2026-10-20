'use client'

import { useMemo } from 'react'
import {
  saveLook as saveHostLook,
  useLook as useHostLook,
  useMediaPreferences,
  type HostLook,
  type HostLookResult,
  type HostLookState,
} from '@automatos/theme-sdk'

import { DEFAULT_LOOK } from '../../lib/looks/presets'
import type { Look, ValidationResult } from '../../lib/looks/types'

/**
 * PRD-261 A-14d — the active look and the saved looks as Play's look editor reads them, through
 * the SDK's `looks` area only (PRD §4.3 item 7): `useLook()` for the state, `saveLook()` to keep
 * one, and `useMediaPreferences()` for reduced motion. The host's look runtime checks every look
 * against Play's own look spec (lib/theme-host/builtin/play-look-bridge.tsx), so a look it hands
 * back is one of Play's (lib/looks/types.ts).
 *
 * Outside a look runtime (a component test of one screen) the editor reads what it read before
 * the port: the default look, already loaded (nothing to wait for), not kept (nothing can be),
 * no reduced motion; each change is refused in the SDK's words and logged, and nothing changes.
 */
export interface PlayLook {
  readonly look: Look
  readonly saved: readonly Look[]
  /** False until the looks have been read; the screens show only their heading until then (M72). */
  readonly loaded: boolean
  /** False once the looks could not be kept: the Looks page says so (M98). */
  readonly persisted: boolean
  /** The user asks for reduced motion: no loops, no glow crossfade, no rise. */
  readonly reducedMotion: boolean
  setLook(look: Look): void
  saveLook(look: Look): ValidationResult
  deleteLook(id: string): void
  renameLook(id: string, name: string): ValidationResult
}

const NO_LOOKS: readonly Look[] = Object.freeze([])

/** A look the host's runtime handed back: it passed Play's look spec, so it is one of Play's. */
function playLook(look: HostLook): Look {
  return look as Look
}

function playResult(result: HostLookResult): ValidationResult {
  return result.ok ? { ok: true, look: playLook(result.look) } : result
}

function missingRuntime(action: string): void {
  console.warn(`[looks] ${action} needs a LookProvider above it; nothing changed`)
}

/** The host's state as the editor reads it; `reducedMotion` from the host's media preferences. */
function playLookOf(host: HostLookState, reducedMotion: boolean): PlayLook {
  const current = host.available ? host.look : null
  const available = current !== null
  const warnIfMissing = (action: string): void => {
    if (!available) missingRuntime(action)
  }
  return {
    look: current === null ? DEFAULT_LOOK : playLook(current),
    saved: available ? host.saved.map(playLook) : NO_LOOKS,
    loaded: host.loaded,
    persisted: host.persisted,
    reducedMotion: available && reducedMotion,
    setLook: (look: Look) => {
      warnIfMissing('setLook')
      host.setLook(look)
    },
    saveLook: (look: Look) => {
      warnIfMissing('saveLook')
      return playResult(saveHostLook(look))
    },
    deleteLook: (id: string) => {
      warnIfMissing('deleteLook')
      host.deleteLook(id)
    },
    renameLook: (id: string, name: string) => {
      warnIfMissing('renameLook')
      return playResult(host.renameLook(id, name))
    },
  }
}

/** The looks, re-read whenever the host's runtime changes them. */
export function usePlayLook(): PlayLook {
  const host = useHostLook()
  const { reducedMotion } = useMediaPreferences()
  return useMemo(() => playLookOf(host, reducedMotion), [host, reducedMotion])
}
