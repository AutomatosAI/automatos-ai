'use client'

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { useLookScope } from '@/components/theme-host/look-scope'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'
import { applyLook, applyMotion, clearLook } from '@/lib/looks/document'
import { DEFAULT_LOOK } from '@/lib/looks/presets'
import { browserStorage, readPrepaintLook, rememberLookScope, type StorageSource } from '@/lib/looks/storage'
import { toneOfGround, type PlayTone } from '@/lib/looks/tone'
import type { Look, ValidationResult } from '@/lib/looks/types'

import { BUILTIN_PLAY_HTML_CLASS } from './play-entry'
import { LookRuntimeBridge } from './play-look-bridge'
import { useLookStore, type LookStore } from './play-look-store'
import { clearPlayToneCookie, writePlayToneCookie } from './play-tone-cookie'

/**
 * PRD-260 K3, K4 — the active look for the whole app, applied only while the
 * style is play. Its variables live on <html>, so every screen follows a
 * change at once; the pre-paint script (play-prepaint.ts) has already
 * set them before first paint, and this takes over once storage is read.
 * Until then, under Play, it paints what the pre-paint script paints
 * (readPrepaintLook), so Play is never without its variables: the account's
 * looks wait for the account and its workspace, which can take a while (the
 * backend asleep: account-look-scope.tsx), and a switch to Play from Studio
 * or Classic ran no pre-paint.
 * Under Play the look's ground sets the tone (written to its cookie and
 * handed up to ThemeProvider's forcedTheme); leaving Play takes every
 * property, the motion marks and the tone cookie off again, and drops the
 * forced tone so the user's own returns.
 *
 * F5: the looks are the signed-in user's in the active workspace
 * (useLookScope, from account-look-scope.tsx); another account in this
 * browser reads its own, and under Play the pre-paint script is pointed at
 * this account's active look for the next page load. M25: a change made in
 * another tab arrives here (./play-look-store.ts), and the tone follows it.
 *
 * PRD-261 A-13c, A-14d: this is the host's look runtime (PA-9), mounted by
 * ThemeLookRuntime (components/theme-host/theme-look-runtime.tsx); it lives
 * with builtin Play's registrations, out of Play's folders, since it reads the
 * host's style and storage. It hands itself to the SDK's `looks` area
 * (./play-look-bridge.tsx) for everything below it, which is how Play's look
 * editor reads it (components/looks/use-play-look.ts).
 */

export interface LookContextValue extends Omit<LookStore, 'scope'> {
  /** The user asks for reduced motion: no loops, no glow crossfade, no rise. */
  readonly reducedMotion: boolean
  /** Under Play: the tone the ground of the look on <html> sets (the pre-paint's until storage is read). Undefined otherwise. */
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
  /** The look to paint; null leaves what is on <html> (the first render, before the pre-paint look is read). */
  readonly look: Look | null
  readonly reducedMotion: boolean
}

/**
 * Under Play, while storage isn't read yet, the look the pre-paint script
 * paints, read after mount (so the server's render and the first one agree);
 * null otherwise. Never the default over a stored look: that would flash it.
 */
function usePrepaintLook(storage: StorageSource, waiting: boolean): Look | null {
  const [prepaint, setPrepaint] = useState<Look | null>(null)
  useEffect(() => {
    setPrepaint(waiting ? readPrepaintLook(storage) : null)
  }, [storage, waiting])
  return prepaint
}

function useLookOnDocument({ active, look, reducedMotion }: DocumentLook): void {
  useEffect(() => {
    const root = document.documentElement
    if (!active) {
      clearLook(root)
      return
    }
    applyMotion(root, reducedMotion)
    if (look !== null) applyLook(root, look)
  }, [active, look, reducedMotion])

  useEffect(() => () => clearLook(document.documentElement), [])
}

/** Under Play, the next page load's pre-paint script paints this account's active look (play-prepaint.ts). */
function useLookPointer(storage: StorageSource, active: boolean, scope: string | null): void {
  useEffect(() => {
    if (active && scope !== null) rememberLookScope(storage, scope)
  }, [storage, active, scope])
}

/** Until a look is painted the server's tone (from its cookie) stands, so a reload never flips it. */
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
  /**
   * PRD-261 A-13c: false while the active theme has no look runtime: nothing is painted or forced,
   * exactly as when the style isn't Play. The host keeps the provider mounted either way, so a style
   * switched in place never remounts the page below it (M69).
   */
  readonly enabled?: boolean
}

export function LookProvider({ children, storage = browserStorage, onToneChange, enabled = true }: LookProviderProps) {
  const store = useLookStore(storage, useLookScope())
  const { scope, loaded, persisted, look, saved, setLook, saveLook, deleteLook, renameLook } = store
  const reducedMotion = usePrefersReducedMotion()
  // PRD-261 A-02b, PA-8: Play's look runs under the theme whose <html> class is Play's.
  const playStyle = useUiStyleOptional()?.activeTheme?.htmlClass === BUILTIN_PLAY_HTML_CLASS
  const active = enabled && playStyle
  const prepaint = usePrepaintLook(storage, active && !loaded)
  const painted = loaded ? look : prepaint
  const tone = active && painted !== null ? toneOfGround(painted.ground) : undefined

  useLookOnDocument({ active, look: painted, reducedMotion })
  useLookTone(active, tone, onToneChange)
  useLookPointer(storage, active, scope)

  const value = useMemo<LookContextValue>(
    () => ({ look, saved, loaded, persisted, reducedMotion, tone, setLook, saveLook, deleteLook, renameLook }),
    [look, saved, loaded, persisted, reducedMotion, tone, setLook, saveLook, deleteLook, renameLook],
  )

  return (
    <LookContext.Provider value={value}>
      <LookRuntimeBridge value={value} enabled={enabled}>
        {children}
      </LookRuntimeBridge>
    </LookContext.Provider>
  )
}
