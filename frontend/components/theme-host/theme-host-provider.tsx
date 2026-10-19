'use client'

/**
 * PRD-261 A-01, A-04a — the theme host, in every build (no build flag). It loads a theme only
 * while the style is an installed theme (`theme:install:<uuid>`): it reads that install from the
 * server, hands the theme's frame to the frame slot and its other registered slots to the
 * placement slots (A-12b), and on any failure falls back to Studio with a notice; in safe mode
 * it loads no theme code at all. When the server names no theme for the install (GET
 * /api/themes/active answers `theme: null`, another install or 404: a stale cookie) it turns
 * off, and Studio shows with no notice.
 * Under an installed theme's style it reports 'loading' from the first render, before its effect
 * starts the load, so the frame slot never mounts the page in the fallback first (M69). Under
 * Classic, Studio and builtin Play (which keeps its compiled path until A-04b) it is off: it loads
 * nothing and puts no status on <html>, so those pages are exactly as without it.
 *
 * The sources and the loader are reached only through import() inside the effect, after the
 * style and safe-mode checks, so safe mode and every other style never evaluate them or the
 * host's module global. The loaded result is kept as a whole object, never by handing the frame
 * component to a state setter: React would call a function passed there as an updater.
 *
 * A-04c: every failure that is the theme's own (failure-ledger.ts says which) is recorded per
 * theme in this browser; the third within ten minutes sends the member back to Studio: the
 * server's choice (PUT /api/themes/choice), the style cookie, this document's style, and a notice
 * of its own that stays after the host turns off. The /safe-mode page is safe mode too: it loads
 * no theme code, and turns safe mode on for the rest of the tab. Safe mode also takes the theme's
 * first paint (the root layout's boot: its classes and prepaint variables) off <html>, so what
 * shows is Studio's look, not only Studio's code.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { useUiStyleOptional, type UiStyleContextValue } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import { writeCookie } from '@/lib/cookies'
import { THEME_BOOT_ELEMENT_ID } from '@/lib/theme-host/boot'
import {
  HTML_CLASS_PATTERN,
  SAFE_MODE_PARAM,
  SAFE_MODE_VALUE,
  THEME_HOST_REASON_ATTR,
  THEME_HOST_STATUS_ATTR,
  THEME_NOTICE_ID,
} from '@/lib/theme-host/constants'
import {
  clearThemeFailures,
  countsTowardRevert,
  recordThemeFailure,
  safeLocalStorage,
  shouldRevert,
} from '@/lib/theme-host/failure-ledger'
import { isInstalledThemeStyle } from '@/lib/theme-host/registry'
import { isSafeModeRequested, safeSessionStorage } from '@/lib/theme-host/safe-mode'
import type { ThemeFailureReason, ThemeHostStatus, ThemeLoadOk, ThemeLoadResult } from '@/lib/theme-host/types'
import {
  STYLE_HTML_CLASS_NAMES,
  styleHtmlClasses,
  UI_STYLE_COOKIE,
  UI_STYLE_COOKIE_MAX_AGE_S,
  type UiStyle,
} from '@/lib/ui-style'
import { ThemeHostContext, type ThemeHostContextValue } from './theme-host-context'

interface HostState {
  readonly status: ThemeHostStatus
  readonly reason: ThemeFailureReason | null
  /** The whole ok result while the theme is active. */
  readonly active: ThemeLoadOk | null
  /** The installed theme's style this state is for; null when off for no style. */
  readonly style: UiStyle | null
}

const OFF_STATE: HostState = { status: 'off', reason: null, active: null, style: null }

function loadingState(style: UiStyle): HostState {
  return { status: 'loading', reason: null, active: null, style }
}

/** A finished load as host state: active, off with no active install (no notice), or failed with its reason. */
function settledState(style: UiStyle, result: ThemeLoadResult): HostState {
  if (result.ok) return { status: 'active', reason: null, active: result, style }
  if (result.reason === null) return { ...OFF_STATE, style }
  return { status: 'failed', reason: result.reason, active: null, style }
}

/**
 * The state as this render sees it. A style the effect has not reached yet (the first render
 * under it, or the render after the style changed) is loading for an installed theme and off for
 * none, never the previous style's state.
 */
function stateForStyle(state: HostState, themeStyle: UiStyle | null): HostState {
  if (state.style === themeStyle) return state
  return themeStyle === null ? OFF_STATE : loadingState(themeStyle)
}

const LOG_PREFIX = '[theme-host]'

export const THEME_FALLBACK_NOTICE = 'The theme could not start, so Studio is showing instead.'
export const THEME_SAFE_MODE_NOTICE = 'Safe mode: themes are off in this tab.'
export const THEME_FAILURE_TEXT: Readonly<Record<ThemeFailureReason, string>> = {
  manifest: 'Its theme.json could not be read.',
  engines: 'It was built for a different theme SDK.',
  load: 'Its code failed to load.',
  register: 'It failed while starting.',
  look: 'Its styles failed to load.',
  render: 'It failed while drawing the page.',
  auth: 'Your session was refused. Sign in again to load it.',
  revoked: 'This version of it was withdrawn.',
  'opt-in': 'You have not opted in to its private code.',
  unavailable: 'Its files could not be reached just now. Reload to try again.',
}

/** A `warn` revocation: the theme runs, and this notice says its version was withdrawn. */
export const THEME_REVOKED_WARNING_NOTICE = 'This theme is still running, but its maintainer withdrew this version.'

/** The revert's notice has its own id, so the host turning off (which dismisses its notice) leaves it. */
export const THEME_REVERT_NOTICE_ID = 'theme-host-revert'
export const THEME_REVERTED_NOTICE = 'This theme failed three times in ten minutes, so you are back on Studio.'
export const THEME_REVERTED_TEXT = 'You can choose it again in Settings once it is fixed.'
/** Where the revert sends the member: the style, and the server choice of the same name. */
const REVERT_STYLE: UiStyle = 'studio'

/** The page that turns safe mode on for its tab (app/safe-mode/page.tsx). */
export const SAFE_MODE_PATH = '/safe-mode'
/** The query that asks for safe mode (PRD-261 §6), as the /safe-mode page asks for it. */
export const SAFE_MODE_QUERY = `?${SAFE_MODE_PARAM}=${SAFE_MODE_VALUE}`

/**
 * Safe mode shows Studio, so it takes the installed theme's first paint off <html>: the root
 * layout's boot (THEME_BOOT_SCRIPT) has already put the theme's classes and its prepaint
 * variables there before any page code runs. Studio's classes go on, every other style's class
 * and the boot's own classes come off, and so does every custom property on <html>'s inline
 * style: in safe mode nothing else writes one there (no theme look runs, and the Play look
 * document runs only under Play).
 */
export const SAFE_MODE_ADD_CLASSES: readonly string[] = styleHtmlClasses(REVERT_STYLE)
export const SAFE_MODE_REMOVE_CLASSES: readonly string[] = STYLE_HTML_CLASS_NAMES.filter(
  (name) => !SAFE_MODE_ADD_CLASSES.includes(name),
)
/** The id of the /safe-mode page's script; its data attributes carry the two lists above. */
export const SAFE_MODE_UNPAINT_ID = 'automatos-safe-mode-unpaint'

/**
 * The same as unpaintThemeBoot, as the one constant script /safe-mode puts first in its HTML, so
 * the page's first paint is Studio's. It is never built from data: the lists come from its own
 * element's data attributes and the boot's JSON element, every class is checked as the boot
 * checks it, and it writes with classList and style.removeProperty only.
 */
export const SAFE_MODE_UNPAINT_SCRIPT = `(function(){
var d=document.documentElement,C=/${HTML_CLASS_PATTERN.source}/,e=document.getElementById("${SAFE_MODE_UNPAINT_ID}"),b=null;
try{var j=document.getElementById("${THEME_BOOT_ELEMENT_ID}");b=j?JSON.parse(j.textContent||"null"):null;}catch(x){b=null;}
function list(n){var v=e?e.getAttribute(n):null;return v?v.split(" "):[];}
var add=list("data-add"),drop=list("data-remove").concat(b!==null&&typeof b==="object"&&Array.isArray(b.addClasses)?b.addClasses:[]);
for(var i=0;i<drop.length;i++){if(typeof drop[i]==="string"&&C.test(drop[i])&&add.indexOf(drop[i])<0){d.classList.remove(drop[i]);}}
for(var k=0;k<add.length;k++){if(C.test(add[k])){d.classList.add(add[k]);}}
var s=d.style,n=[];for(var m=0;m<s.length;m++){if(s[m].indexOf("--")===0){n.push(s[m]);}}
for(var p=0;p<n.length;p++){s.removeProperty(n[p]);}
})();`

/** The classes the root layout's boot put on <html>, read from its JSON element; none if unreadable. */
function bootAddedClasses(): string[] {
  try {
    const boot: unknown = JSON.parse(document.getElementById(THEME_BOOT_ELEMENT_ID)?.textContent || 'null')
    const added = boot !== null && typeof boot === 'object' ? (boot as { addClasses?: unknown }).addClasses : null
    return Array.isArray(added) ? added.filter((name): name is string => typeof name === 'string') : []
  } catch {
    return []
  }
}

/** Takes the installed theme's first paint off <html> and puts Studio's on (see SAFE_MODE_ADD_CLASSES). */
export function unpaintThemeBoot(): void {
  const root = document.documentElement
  const drop = [...SAFE_MODE_REMOVE_CLASSES, ...bootAddedClasses()]
  drop
    .filter((name) => HTML_CLASS_PATTERN.test(name) && !SAFE_MODE_ADD_CLASSES.includes(name))
    .forEach((name) => root.classList.remove(name))
  SAFE_MODE_ADD_CLASSES.forEach((name) => root.classList.add(name))
  const custom = Array.from(root.style).filter((name) => name.startsWith('--'))
  custom.forEach((name) => root.style.removeProperty(name))
}

/** Turns safe mode on for this tab, as `?safe_mode=themes` does; the /safe-mode page calls it. */
export function requestSafeModeForThisTab(): void {
  isSafeModeRequested(SAFE_MODE_QUERY, safeSessionStorage())
}

/** Safe mode here: asked for by the query, by being the /safe-mode page, or by an earlier page in this tab. */
function isSafeModeHere(): boolean {
  const { pathname, search } = window.location
  return isSafeModeRequested(pathname === SAFE_MODE_PATH ? SAFE_MODE_QUERY : search, safeSessionStorage())
}

function failedStart(reason: ThemeFailureReason, error: unknown): ThemeLoadResult {
  console.error(LOG_PREFIX, reason, error)
  return { ok: false, reason, detail: error instanceof Error ? error.message : String(error) }
}

/**
 * The sources chunk, fetched on demand. A chunk that fails to load is the host's own chunk (a
 * deploy, the network), not the theme's: 'unavailable', which never counts toward the revert.
 */
async function startTheme(style: UiStyle): Promise<ThemeLoadResult> {
  const chunk = await import('@/lib/theme-host/sources').then(
    (sources) => ({ ok: true as const, sources }),
    (error: unknown) => ({ ok: false as const, error }),
  )
  if (!chunk.ok) return failedStart('unavailable', chunk.error)
  try {
    return await chunk.sources.loadThemeForStyle(style)
  } catch (error) {
    return failedStart('load', error)
  }
}

/**
 * Loads run one after another. Every load writes the same class, stylesheet and variables on
 * <html>, so a cancelled load must settle and clean up before the next one starts; otherwise its
 * late cleanup would remove the live load's look and undo its registration. A load cancelled
 * while it waits its turn never starts.
 */
let loadQueue: Promise<void> = Promise.resolve()

function enqueueLoad(run: () => Promise<void>): void {
  loadQueue = loadQueue.then(run).catch((error: unknown) => console.error(LOG_PREFIX, 'load', error))
}

/** Loads the theme of `themeStyle` (an installed theme's style), or stays off for null. */
function useThemeLoad(themeStyle: UiStyle | null) {
  const [state, setState] = useState<HostState>(OFF_STATE)
  const activeRef = useRef<ThemeLoadOk | null>(null)

  useEffect(() => {
    if (themeStyle === null) {
      setState(OFF_STATE)
      return undefined
    }
    if (isSafeModeHere()) {
      unpaintThemeBoot()
      setState({ status: 'safe-mode', reason: null, active: null, style: themeStyle })
      return undefined
    }
    let cancelled = false
    setState(loadingState(themeStyle))
    enqueueLoad(async () => {
      if (cancelled) return
      const result = await startTheme(themeStyle)
      if (cancelled) {
        if (result.ok) result.cleanup()
        return
      }
      if (result.ok) activeRef.current = result
      setState(settledState(themeStyle, result))
    })
    return () => {
      cancelled = true
      activeRef.current?.cleanup()
      activeRef.current = null
    }
  }, [themeStyle])

  const reportRenderFailure = useCallback((error: unknown) => {
    const active = activeRef.current
    if (active === null) return
    console.error(LOG_PREFIX, 'render', error)
    activeRef.current = null
    active.cleanup()
    setState((previous) => ({ ...previous, status: 'failed', reason: 'render', active: null }))
  }, [])

  return { state: stateForStyle(state, themeStyle), reportRenderFailure }
}

function clearThemeHostAttributes(): void {
  document.documentElement.removeAttribute(THEME_HOST_STATUS_ATTR)
  document.documentElement.removeAttribute(THEME_HOST_REASON_ATTR)
}

/**
 * Mirrors the status and reason onto <html>, for the browser lane to wait on. Off writes
 * nothing: a page with no theme style carries no trace of the host.
 */
function useThemeHostAttributes(status: ThemeHostStatus, reason: ThemeFailureReason | null): void {
  useEffect(() => {
    if (status === 'off') {
      clearThemeHostAttributes()
      return
    }
    const root = document.documentElement
    root.setAttribute(THEME_HOST_STATUS_ATTR, status)
    if (reason === null) root.removeAttribute(THEME_HOST_REASON_ATTR)
    else root.setAttribute(THEME_HOST_REASON_ATTR, reason)
  }, [status, reason])

  useEffect(() => clearThemeHostAttributes, [])
}

/**
 * The one notice: the fallback with its reason, or safe mode; gone again once the theme is off or
 * active, except for an active theme whose version has a `warn` revocation (`warning` is its reason).
 */
function useThemeNotice(status: ThemeHostStatus, reason: ThemeFailureReason | null, warning: string | null): void {
  useEffect(() => {
    if (status === 'active' && warning !== null) {
      const description = warning || THEME_FAILURE_TEXT.revoked
      toast.warning(THEME_REVOKED_WARNING_NOTICE, { id: THEME_NOTICE_ID, duration: Infinity, description })
      toast.dismiss(THEME_REVERT_NOTICE_ID)
    } else if (status === 'failed' && reason !== null) {
      toast.error(THEME_FALLBACK_NOTICE, { id: THEME_NOTICE_ID, duration: Infinity, description: THEME_FAILURE_TEXT[reason] })
    } else if (status === 'safe-mode') {
      toast(THEME_SAFE_MODE_NOTICE, { id: THEME_NOTICE_ID, duration: Infinity })
    } else if (status === 'off') {
      toast.dismiss(THEME_NOTICE_ID)
    } else if (status === 'active') {
      toast.dismiss(THEME_NOTICE_ID)
      toast.dismiss(THEME_REVERT_NOTICE_ID)
    }
  }, [status, reason, warning])
}

type SetStyle = UiStyleContextValue['setStyle']

/**
 * Sends the member back to Studio: the server's choice (every tab and device agrees), the style
 * cookie (the next request paints Studio), this document's style, and the notice. The theme's
 * failures are forgotten once the server took the choice; when it refused, they stay, so the
 * theme's next failure tries again.
 */
async function revertToStudio(theme: UiStyle, setStyle: SetStyle | null): Promise<void> {
  writeCookie(UI_STYLE_COOKIE, REVERT_STYLE, UI_STYLE_COOKIE_MAX_AGE_S)
  setStyle?.(REVERT_STYLE)
  toast.error(THEME_REVERTED_NOTICE, { id: THEME_REVERT_NOTICE_ID, duration: Infinity, description: THEME_REVERTED_TEXT })
  try {
    await apiClient.putThemeChoice(REVERT_STYLE)
    clearThemeFailures(theme, safeLocalStorage())
  } catch (error) {
    console.error(LOG_PREFIX, 'the Studio choice was not saved', error)
  }
}

/**
 * A-04c: records each failure that counts, once per failed state (an effect that runs again for
 * the same state, as React's strict mode does, records nothing more), and reverts to Studio on
 * the third within the window.
 */
function useFailureRevert(state: HostState, setStyle: SetStyle | null): void {
  const recorded = useRef<HostState | null>(null)
  useEffect(() => {
    if (state.status !== 'failed' || state === recorded.current) return
    recorded.current = state
    const { reason, style } = state
    if (reason === null || style === null || !countsTowardRevert(reason)) return
    if (shouldRevert(recordThemeFailure(style, Date.now(), safeLocalStorage()))) void revertToStudio(style, setStyle)
  }, [state, setStyle])
}

/** The style the host loads a theme for: an installed theme's, else null (off). */
function installedThemeStyle(uiStyle: UiStyleContextValue | null): UiStyle | null {
  const style = uiStyle?.style ?? null
  return style !== null && isInstalledThemeStyle(style) ? style : null
}

/** PRD-261 A-04a: in every build; it loads a theme only under an installed theme's style. */
export function ThemeHostProvider({ children }: { children: ReactNode }) {
  const uiStyle = useUiStyleOptional()
  const { state, reportRenderFailure } = useThemeLoad(installedThemeStyle(uiStyle))
  useThemeHostAttributes(state.status, state.reason)
  useThemeNotice(state.status, state.reason, state.active?.revoked?.reason ?? null)
  useFailureRevert(state, uiStyle?.setStyle ?? null)
  const value = useMemo<ThemeHostContextValue>(
    () => ({
      status: state.status,
      reason: state.reason,
      frame: state.active?.frame ?? null,
      slots: state.active?.slots ?? null,
      pages: state.active?.pages ?? null,
      routes: state.active?.routes ?? null,
      reportRenderFailure,
    }),
    [state, reportRenderFailure],
  )
  return <ThemeHostContext.Provider value={value}>{children}</ThemeHostContext.Provider>
}
