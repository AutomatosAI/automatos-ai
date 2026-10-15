'use client'

import { createContext, useContext, useEffect, useMemo, type ReactNode } from 'react'

import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'
import { applyLook, applyMotion, clearLook } from '@/lib/looks/document'
import { DEFAULT_LOOK } from '@/lib/looks/presets'
import { browserStorage, rememberLookScope, type StorageSource } from '@/lib/looks/storage'
import { clearPlayToneCookie, toneOfGround, writePlayToneCookie, type PlayTone } from '@/lib/looks/tone'
import type { Look, ValidationResult } from '@/lib/looks/types'

import { useLookScope } from './look-scope'
import { useLookStore, type LookStore } from './use-look-store'

/**
 * PRD-260 K3, K4 — the active look for the whole app, applied only while the
 * style is play. Its variables live on <html>, so every screen follows a
 * change at once; the pre-paint script (lib/looks/prepaint.ts) has already
 * set them before first paint, and this takes over once storage is read.
 * Under Play the look's ground sets the tone (written to its cookie and
 * handed up to ThemeProvider's forcedTheme); leaving Play takes every
 * property, the motion marks and the tone cookie off again, and drops the
 * forced tone so the user's own returns.
 *
 * F5: the looks are the signed-in user's in the active workspace
 * (useLookScope, from account-look-scope.tsx); another account in this
 * browser reads its own, and under Play the pre-paint script is pointed at
 * this account's active look for the next page load. M25: a change made in
 * another tab arrives here (use-look-store.ts), and the tone follows it.
 */

export interface LookContextValue extends Omit<LookStore, 'scope'> {
  /** The user asks for reduced motion: no loops, no glow crossfade, no rise. */
  readonly reducedMotion: boolean
  /** Under Play, once storage is read: the tone the look's ground sets. Undefined otherwise. */
  readonly tone: PlayTone | undefined
}

function missingProvider(action: string): void {
  console.warn(`[looks] ${action} needs a LookProvider above it; nothing changed`)
}

const NO_PROVIDER: ValidationResult = { ok: false, errors: ["Looks aren't available here."] }

// Outside a provider (a component test of a single screen) everything reads as the default look,
// already loaded (nothing to wait for) and not kept (nothing can be).
const LookContext = createContext<LookContextValue>({
  look: DEFAULT_LOOK,
  saved: [],
  loaded: true,
  persisted: false,
  reducedMotion: false,
  tone: undefined,
  setLook: () => missingProvider('setLook'),
  saveLook: () => {
    missingProvider('saveLook')
    return NO_PROVIDER
  },
  deleteLook: () => missingProvider('deleteLook'),
  renameLook: () => {
    missingProvider('renameLook')
    return NO_PROVIDER
  },
})

export function useLook(): LookContextValue {
  return useContext(LookContext)
}

interface DocumentLook {
  readonly active: boolean
  readonly loaded: boolean
  readonly look: Look
  readonly reducedMotion: boolean
}

/** Until storage is read the pre-paint values stand: applying the default first would flash it. */
function useLookOnDocument({ active, loaded, look, reducedMotion }: DocumentLook): void {
  useEffect(() => {
    const root = document.documentElement
    if (!active) {
      clearLook(root)
      return
    }
    applyMotion(root, reducedMotion)
    if (loaded) applyLook(root, look)
  }, [active, loaded, look, reducedMotion])

  useEffect(() => () => clearLook(document.documentElement), [])
}

/** Under Play, the next page load's pre-paint script paints this account's active look (lib/looks/prepaint.ts). */
function useLookPointer(storage: StorageSource, active: boolean, scope: string | null): void {
  useEffect(() => {
    if (active && scope !== null) rememberLookScope(storage, scope)
  }, [storage, active, scope])
}

/** Until storage is read the server's tone (from its cookie) stands, so a reload never flips it. */
function useLookTone(active: boolean, tone: PlayTone | undefined, onToneChange?: (tone: PlayTone | undefined) => void): void {
  useEffect(() => {
    if (!active) {
      clearPlayToneCookie()
      onToneChange?.(undefined)
      return
    }
    if (tone === undefined) return
    writePlayToneCookie(tone)
    onToneChange?.(tone)
  }, [active, tone, onToneChange])
}

export interface LookProviderProps {
  readonly children: ReactNode
  /** Tests pass their own (or one that throws, for blocked storage). */
  readonly storage?: StorageSource
  /** Receives the tone Play forces (undefined when it forces none): components/providers.tsx hands it to ThemeProvider. */
  readonly onToneChange?: (tone: PlayTone | undefined) => void
}

export function LookProvider({ children, storage = browserStorage, onToneChange }: LookProviderProps) {
  const store = useLookStore(storage, useLookScope())
  const { scope, loaded, persisted, look, saved, setLook, saveLook, deleteLook, renameLook } = store
  const reducedMotion = usePrefersReducedMotion()
  const active = useUiStyleOptional()?.isPlay ?? false
  const tone = active && loaded ? toneOfGround(look.ground) : undefined

  useLookOnDocument({ active, loaded, look, reducedMotion })
  useLookTone(active, tone, onToneChange)
  useLookPointer(storage, active, scope)

  const value = useMemo<LookContextValue>(
    () => ({ look, saved, loaded, persisted, reducedMotion, tone, setLook, saveLook, deleteLook, renameLook }),
    [look, saved, loaded, persisted, reducedMotion, tone, setLook, saveLook, deleteLook, renameLook],
  )

  return <LookContext.Provider value={value}>{children}</LookContext.Provider>
}
