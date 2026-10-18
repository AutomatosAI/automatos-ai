'use client'

/**
 * PRD-261 A-01, A-04a — the theme host, in every build (no build flag). It loads a theme only
 * while the style is an installed theme (`theme:install:<uuid>`): it reads that install from the
 * server, hands the theme's frame to the frame slot, and on any failure falls back to Studio with
 * a notice; in safe mode it loads no theme code at all. When the server has no active install (GET
 * /api/themes/active answers 404: a stale cookie) it turns off, and Studio shows with no notice.
 * Under an installed theme's style it reports 'loading' from the first render, before its effect
 * starts the load, so the frame slot never mounts the page in the fallback first (M69). Under
 * Classic, Studio and builtin Play (which keeps its compiled path until A-04b) it is off: it loads
 * nothing and puts no status on <html>, so those pages are exactly as without it.
 *
 * The sources and the loader are reached only through import() inside the effect, after the
 * style and safe-mode checks, so safe mode and every other style never evaluate them or the
 * host's module global. The loaded result is kept as a whole object, never by handing the frame
 * component to a state setter: React would call a function passed there as an updater.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { THEME_HOST_REASON_ATTR, THEME_HOST_STATUS_ATTR, THEME_NOTICE_ID } from '@/lib/theme-host/constants'
import { isInstalledThemeStyle } from '@/lib/theme-host/registry'
import { isSafeModeRequested, safeSessionStorage } from '@/lib/theme-host/safe-mode'
import type { ThemeFailureReason, ThemeHostStatus, ThemeLoadOk, ThemeLoadResult } from '@/lib/theme-host/types'
import type { UiStyle } from '@/lib/ui-style'
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
}

/** The sources chunk, fetched on demand; a chunk that fails to load is a 'load' failure. */
async function startTheme(style: UiStyle): Promise<ThemeLoadResult> {
  try {
    const { loadThemeForStyle } = await import('@/lib/theme-host/sources')
    return await loadThemeForStyle(style)
  } catch (error) {
    console.error(LOG_PREFIX, 'load', error)
    return { ok: false, reason: 'load', detail: error instanceof Error ? error.message : String(error) }
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
    if (isSafeModeRequested(window.location.search, safeSessionStorage())) {
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

/** The one notice: the fallback with its reason, or safe mode; gone again once the theme is off or active. */
function useThemeNotice(status: ThemeHostStatus, reason: ThemeFailureReason | null): void {
  useEffect(() => {
    if (status === 'failed' && reason !== null) {
      toast.error(THEME_FALLBACK_NOTICE, { id: THEME_NOTICE_ID, duration: Infinity, description: THEME_FAILURE_TEXT[reason] })
    } else if (status === 'safe-mode') {
      toast(THEME_SAFE_MODE_NOTICE, { id: THEME_NOTICE_ID, duration: Infinity })
    } else if (status === 'off' || status === 'active') {
      toast.dismiss(THEME_NOTICE_ID)
    }
  }, [status, reason])
}

/** The style the host loads a theme for: an installed theme's, else null (off). */
function useInstalledThemeStyle(): UiStyle | null {
  const style = useUiStyleOptional()?.style ?? null
  return style !== null && isInstalledThemeStyle(style) ? style : null
}

/** PRD-261 A-04a: in every build; it loads a theme only under an installed theme's style. */
export function ThemeHostProvider({ children }: { children: ReactNode }) {
  const { state, reportRenderFailure } = useThemeLoad(useInstalledThemeStyle())
  useThemeHostAttributes(state.status, state.reason)
  useThemeNotice(state.status, state.reason)
  const value = useMemo<ThemeHostContextValue>(
    () => ({ status: state.status, reason: state.reason, frame: state.active?.frame ?? null, reportRenderFailure }),
    [state, reportRenderFailure],
  )
  return <ThemeHostContext.Provider value={value}>{children}</ThemeHostContext.Provider>
}
