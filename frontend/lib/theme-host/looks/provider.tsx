'use client'

/**
 * PRD-261 A-19d2 (plan §2.2.4) — the generic look runtime for an installed theme with looks: the
 * host mounts it (components/theme-host/theme-look-runtime.tsx) for every style, in one tree shape
 * (M69), and switches it on while the host loads or runs an installed theme that declares looks.
 *
 * Switched on, it reads the theme's look rules from the page's boot (./page-boot.ts: the same
 * prepaint table, guard and presets the boot script painted the first frame from), or, when the
 * page carries the base boot (the server's boot fetch failed), from the install's boot the browser
 * fetches once per install (../boot-client.ts), painted through the same paint path; keeps the
 * signed-in user's looks in this workspace (./store.ts, the scope given), paints the active look on
 * <html> (./paint.ts) with the reduced-motion marks (./motion.ts), forces the tone the theme's tone
 * rule gives the look (./tone-cookie.ts: its cookie, and `onToneChange` for ThemeProvider's
 * forcedTheme) and hands the looks to the SDK (./bridge.tsx). Until the account's looks are read it
 * paints the look the boot script painted (readPrepaintLook), so the theme is never without its
 * variables, also while the workspace is still unknown; once they are read it points the next
 * load's boot script at this account's active look (rememberLookScope, F5).
 *
 * Switched off, it paints nothing, forces no tone and deletes its tone cookie. `keepFirstPaint`
 * (the host loads or runs an installed theme) leaves <html> as the boot painted it; otherwise (a
 * failed load, safe mode, every other style) the table's variables and the motion marks come off.
 * The tone goes back to the user when no look runtime runs at all (the host's ThemeLookRuntime),
 * so this runtime never hands up "no tone" over another runtime's.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'

import { pageInstalledBoot, type InstalledBootSource } from '../boot-client'
import type { PrepaintTable, ThemeTone } from '../boot-types'
import type { ThemeTonePolicy } from '../registry-types'
import { isSafeModeHere } from '../safe-mode'
import { LookRuntimeBridge, type LookBridgeSource } from './bridge'
import { applyThemeMotion, clearThemeMotion } from './motion'
import { bootLookRules, bootNamesTheme, readPageBoot } from './page-boot'
import { applyPaint, clearPaint } from './paint'
import type { LookServer } from './server'
import type { ThemeLook, ThemeLookRules } from './spec'
import { browserStorage, lookTheme, readPrepaintLook, rememberLookScope, type LookTheme, type StorageSource } from './storage'
import { useLookStore } from './store'
import { clearThemeToneCookie, toneOfLook, writeThemeToneCookie } from './tone-cookie'

/** The policy under which the look sets the tone. */
const FROM_LOOK: ThemeTonePolicy = 'from-look'

/** Rules not read yet (undefined), none for this theme (null), or the theme's. */
type BootRules = ThemeLookRules | null | undefined

interface KeptRules {
  readonly themeId: string
  readonly rules: ThemeLookRules | null
}

/**
 * The look rules of `themeId`'s boot, read after mount, once per theme id: one rules object per
 * boot, which the store compares by reference. The page's boot when it names a theme; a page whose
 * boot names none (the base boot) takes the install's boot from `bootSource`. Undefined until read.
 */
function useBootRules(themeId: string | null, installId: string | null, bootSource: InstalledBootSource): BootRules {
  const [kept, setKept] = useState<KeptRules | null>(null)
  useEffect(() => {
    if (themeId === null || kept?.themeId === themeId) return undefined
    const page = readPageBoot()
    // Safe mode asks for no theme code: the page's own boot (a base boot there) is all it reads.
    if (installId === null || bootNamesTheme(page) || isSafeModeHere()) {
      setKept({ themeId, rules: bootLookRules(page, themeId) })
      return undefined
    }
    let live = true
    void bootSource(installId).then((boot) => {
      if (live) setKept({ themeId, rules: bootLookRules(boot, themeId) })
    })
    return () => {
      live = false
    }
  }, [themeId, installId, bootSource, kept])
  if (themeId === null) return null
  return kept?.themeId === themeId ? kept.rules : undefined
}

/** The look the boot script paints, read after mount while the account's looks aren't read; null otherwise. */
function usePrepaintLook(storage: StorageSource, theme: LookTheme | null, waiting: boolean): ThemeLook | null {
  const [prepaint, setPrepaint] = useState<ThemeLook | null>(null)
  useEffect(() => {
    setPrepaint(waiting && theme !== null ? readPrepaintLook(storage, theme) : null)
  }, [storage, theme, waiting])
  return prepaint
}

interface DocumentLook {
  /** The table of the look painted, or held for clearing; null when the theme has none. */
  readonly table: PrepaintTable | null
  /** The look to paint; null while there is none (the runtime off, or before the boot's look is read). */
  readonly look: ThemeLook | null
  readonly active: boolean
  readonly reducedMotion: boolean
  readonly keepFirstPaint: boolean
}

/** The last table this runtime painted with or held, until its variables come off. */
interface HeldTable {
  current: PrepaintTable | null
}

/** Takes the held table's variables off `root` once: a later run never takes off what another runtime painted since. */
function clearHeldPaint(root: HTMLElement, held: HeldTable): void {
  if (held.current === null) return
  clearPaint(root, held.current)
  held.current = null
}

/**
 * Switched off, the motion marks come off and, unless the host still loads or runs an installed theme
 * (keepFirstPaint), the variables of the last table this runtime held. A theme left in place (for
 * Studio or Classic: only a switch TO an installed theme reloads) hands in no table, as its rules go
 * with its id, and the host's status can trail the style by a commit; so the table is
 * held, not replaced by none, into the commit where keepFirstPaint turns false.
 */
function useLookOnDocument({ table, look, active, reducedMotion, keepFirstPaint }: DocumentLook): void {
  const held = useRef<PrepaintTable | null>(null)
  useEffect(() => {
    const root = document.documentElement
    if (table !== null) held.current = table
    if (!active) {
      clearThemeMotion(root)
      if (!keepFirstPaint) clearHeldPaint(root, held)
      return
    }
    applyThemeMotion(root, reducedMotion)
    if (look !== null && table !== null) applyPaint(root, table, look)
  }, [table, look, active, reducedMotion, keepFirstPaint])

  useEffect(() => () => {
    const root = document.documentElement
    clearThemeMotion(root)
    clearHeldPaint(root, held)
  }, [])
}

interface LookTone {
  /** The theme whose look tone the cookie keeps (its theme.json id); null for every other style. */
  readonly themeId: string | null
  readonly active: boolean
  /** The rules are not read yet: nothing is said about the tone (the server's stands). */
  readonly pending: boolean
  /** The theme's tone follows its look and its table carries a tone rule. */
  readonly follows: boolean
  readonly tone: ThemeTone | null
  readonly onToneChange?: (tone: ThemeTone | undefined) => void
}

/** Until a look is painted the server's tone (from its cookie) stands, so a reload never flips it. */
function useLookTone({ themeId, active, pending, follows, tone, onToneChange }: LookTone): void {
  useEffect(() => {
    if (pending) return
    if (!active || !follows || themeId === null) {
      clearThemeToneCookie()
      return
    }
    if (tone === null) return
    writeThemeToneCookie(themeId, tone)
    onToneChange?.(tone)
  }, [themeId, active, pending, follows, tone, onToneChange])
}

/** The next page load's boot script paints this account's active look (F5). */
function useLookPointer(storage: StorageSource, theme: LookTheme | null, scope: string | null): void {
  useEffect(() => {
    if (theme !== null && scope !== null) rememberLookScope(storage, scope, theme.keys)
  }, [storage, theme, scope])
}

export interface ThemeLookProviderProps {
  readonly children: ReactNode
  /** The installed theme with looks that is the style (its theme.json id); null for every other style. */
  readonly themeId: string | null
  /**
   * The install that theme runs as (its uuid): its boot is fetched when the page's boot names no
   * theme. Null (the default) never fetches: the page's boot alone.
   */
  readonly installId?: string | null
  /** Where that boot comes from; tests pass their own. Default: GET /api/themes/boot/{installId}, once per install. */
  readonly bootSource?: InstalledBootSource
  /** The host loads or runs that theme: only then do its looks run. */
  readonly enabled: boolean
  /** The theme's tone policy: under 'from-look' its tone rule sets the tone. */
  readonly tonePolicy: ThemeTonePolicy
  /** Whose looks: the signed-in user in this workspace (AccountLookScope); null until both are known. */
  readonly scope: string | null
  /** Receives the tone the look forces: components/providers.tsx hands it to ThemeProvider. */
  readonly onToneChange?: (tone: ThemeTone | undefined) => void
  /** Switched off, leave <html> as the root layout's boot painted it (the host loads or runs an installed theme). */
  readonly keepFirstPaint?: boolean
  /** Tests pass their own (or one that throws, for blocked storage). */
  readonly storage?: StorageSource
  /** Where the looks are kept besides this browser; tests pass their own or null. Default: the API. */
  readonly server?: LookServer | null
}

export function ThemeLookProvider(props: ThemeLookProviderProps) {
  const { children, themeId, enabled, tonePolicy, scope, onToneChange, keepFirstPaint = false, storage = browserStorage } = props
  const rules = useBootRules(themeId, props.installId ?? null, props.bootSource ?? pageInstalledBoot)
  const active = enabled && themeId !== null && rules !== null && rules !== undefined
  const theme = useMemo(() => (active && themeId !== null && rules ? lookTheme(themeId, rules) : null), [active, themeId, rules])
  const store = useLookStore(storage, scope, theme, { server: props.server })
  const prepaint = usePrepaintLook(storage, theme, active && !store.loaded)
  const painted = store.loaded ? store.look : prepaint
  const table = rules?.table ?? null
  const follows = tonePolicy === FROM_LOOK && table?.tone !== undefined
  const tone = active && follows && painted !== null ? toneOfLook(table?.tone, painted) : null
  const reducedMotion = usePrefersReducedMotion()

  useLookOnDocument({ table, look: painted, active, reducedMotion, keepFirstPaint })
  useLookTone({ themeId, active, pending: enabled && rules === undefined, follows, tone, onToneChange })
  useLookPointer(storage, theme, store.scope)

  const source = useMemo<LookBridgeSource | null>(() => (active && rules ? { store, rules, tone } : null), [active, rules, store, tone])
  return <LookRuntimeBridge source={source}>{children}</LookRuntimeBridge>
}
