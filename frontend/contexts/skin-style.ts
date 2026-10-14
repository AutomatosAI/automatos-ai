'use client'

/**
 * PRD-259 §4 — the skin half of the Style axis, as the hooks UiStyleProvider composes.
 *
 * The style is Classic, Studio or `skin:<id>`; the base is the design system the page's
 * layouts follow (a skin's own `base`), so a skin never changes a layout. Choosing a skin
 * resolves it from the installed list through the one validator and the one stylesheet
 * writer; when it isn't there, fails either, or can't be written, the base style shows, the
 * cookie goes back to the base, the cache is cleared and a quiet notice says so once.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { useInstalledSkins, type InstalledSkins } from '@/hooks/use-installed-skins'
import { applySkinToDocument, removeSkinFromDocument } from '@/lib/skins/document'
import { registerSkinFonts } from '@/lib/skins/fonts'
import { lockedToneOf, resolveInstalledSkin, type ResolvedSkin } from '@/lib/skins/installed'
import type { SkinTone } from '@/lib/skins/schema'
import {
  STUDIO_HTML_CLASS,
  UI_SKIN_BASE_COOKIE,
  UI_SKIN_TONE_COOKIE,
  UI_STYLE_COOKIE,
  UI_STYLE_COOKIE_MAX_AGE_S,
  UI_STYLE_STORAGE_KEY,
  isBaseStyle,
  isSkinStyle,
  skinIdOf,
  skinStyle,
  type BaseStyle,
  type UiStyle,
} from '@/lib/ui-style'

export const BASE_STYLE_LABELS: Readonly<Record<BaseStyle, string>> = { classic: 'Classic', studio: 'Studio' }
const UNNAMED_SKIN = 'The skin'
/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0

/** The fallback notice (PRD-259 §4): '<name> isn't available here, so <base> is showing.' */
export function fallbackNotice(name: string | null, base: BaseStyle): string {
  return `${name ?? UNNAMED_SKIN} isn't available here, so ${BASE_STYLE_LABELS[base]} is showing.`
}

function writeCookie(name: string, value: string, maxAgeS = UI_STYLE_COOKIE_MAX_AGE_S): void {
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${name}=${value}; Path=/; Max-Age=${maxAgeS}; SameSite=Lax${secure}`
}

/**
 * Where the next request (the cookies) and this document (the studio class) see the style.
 * The base cookie travels with every choice, so the server never needs the database to know
 * a skin's base (decision K1); the tone cookie carries a single-tone skin's lock the same way
 * and is cleared for everything else. Never touches the user's own stored tone.
 */
export function applyUiStyleToDocument(style: UiStyle, base: BaseStyle, lockedTone?: SkinTone): void {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle(STUDIO_HTML_CLASS, base === 'studio')
  writeCookie(UI_STYLE_COOKIE, style)
  writeCookie(UI_SKIN_BASE_COOKIE, base)
  if (lockedTone) writeCookie(UI_SKIN_TONE_COOKIE, lockedTone)
  else writeCookie(UI_SKIN_TONE_COOKIE, '', COOKIE_EXPIRED_S)
  try {
    window.localStorage.setItem(UI_STYLE_STORAGE_KEY, style)
  } catch {
    // Storage blocked (private mode): the cookies alone carry the choice.
  }
}

interface StyleState {
  readonly style: UiStyle
  readonly base: BaseStyle
  readonly active: ResolvedSkin | null
}

function sameSkin(a: ResolvedSkin | null, b: ResolvedSkin): boolean {
  return a !== null && a.id === b.id && a.css === b.css && a.skin.version === b.skin.version
    && JSON.stringify(a.skin.fonts ?? null) === JSON.stringify(b.skin.fonts ?? null)
}

/** The state a resolved skin makes; an unchanged skin keeps its object (fonts stay registered). */
function skinState(resolved: ResolvedSkin, previous: ResolvedSkin | null): StyleState {
  const active = previous !== null && sameSkin(previous, resolved) ? previous : resolved
  return { style: skinStyle(resolved.id), base: resolved.skin.base, active }
}

function baseState(base: BaseStyle): StyleState {
  return { style: base, base, active: null }
}

function useFallbackNotice(): (skinId: string, name: string | null, base: BaseStyle) => void {
  const shown = useRef<ReadonlySet<string>>(new Set())
  return useCallback((skinId: string, name: string | null, base: BaseStyle) => {
    if (shown.current.has(skinId)) return
    shown.current = new Set([...shown.current, skinId])
    toast.info(fallbackNotice(name, base))
  }, [])
}

export interface SkinStyle extends StyleState {
  readonly setStyle: (next: UiStyle) => void
  readonly installed: InstalledSkins
}

export function useSkinStyle(initialStyle: UiStyle, initialBase: BaseStyle): SkinStyle {
  const [state, setState] = useState<StyleState>(() => ({ style: initialStyle, base: initialBase, active: null }))
  const stateRef = useRef(state)
  const installed = useInstalledSkins(isSkinStyle(initialStyle))
  const { skinsRef } = installed
  const notify = useFallbackNotice()

  const commit = useCallback((next: StyleState) => {
    stateRef.current = next
    setState(next)
    applyUiStyleToDocument(next.style, next.base, lockedToneOf(next.active?.skin))
    if (next.active) applySkinToDocument(next.active)
    else removeSkinFromDocument()
  }, [])

  const chooseSkin = useCallback((skinId: string) => {
    const item = skinsRef.current.find((skin) => skin.id === skinId) ?? null
    const resolved = item ? resolveInstalledSkin(item) : null
    if (resolved) {
      commit(skinState(resolved, stateRef.current.active))
      return
    }
    const base = stateRef.current.base
    commit(baseState(base))
    notify(skinId, item?.name ?? null, base)
  }, [commit, notify, skinsRef])

  const setStyle = useCallback((next: UiStyle) => {
    const skinId = skinIdOf(next)
    if (skinId !== null) chooseSkin(skinId)
    else if (isBaseStyle(next)) commit(baseState(next))
  }, [chooseSkin, commit])

  // The list arrived (or changed): repair the cached stylesheet, or fall back.
  useEffect(() => {
    if (installed.status !== 'ready') return
    const skinId = skinIdOf(stateRef.current.style)
    if (skinId !== null) chooseSkin(skinId)
  }, [installed.status, installed.skins, chooseSkin])

  return { ...state, setStyle, installed }
}

/** Registers the active skin's own woff2 faces, and removes them when it changes. */
export function useSkinFonts(active: ResolvedSkin | null): void {
  useEffect(() => {
    if (active === null) return undefined
    const controller = new AbortController()
    let remove: (() => void) | null = null
    void registerSkinFonts(active.skin, active.id, controller.signal).then((cleanup) => {
      if (controller.signal.aborted) cleanup()
      else remove = cleanup
    })
    return () => {
      controller.abort()
      remove?.()
    }
  }, [active])
}

/**
 * Tells the tone owner (ThemeProvider's forcedTheme) which tone a single-tone skin locks. Until
 * the installed list resolves the skin (or when it fails to load), the server's lock from the
 * tone cookie holds, so a reload never paints the base's other tone first.
 */
export function useToneLock(
  { style, active }: Pick<SkinStyle, 'style' | 'active'>,
  cookieTone: SkinTone | undefined,
  onToneLock?: (tone: SkinTone | undefined) => void,
): void {
  const pending = active === null && isSkinStyle(style)
  const tone = pending ? cookieTone : lockedToneOf(active?.skin)
  useEffect(() => {
    onToneLock?.(tone)
  }, [tone, onToneLock])
}
