/**
 * PRD-261 A-02c (PA-12) — the theme choice: the server holds it, the `automatos-style` cookie is
 * its first-paint cache.
 *
 * - Choosing (the theme menu, ⌘K): `useChooseStyle` stores the pick with `PUT /api/themes/choice`
 *   ('classic' | 'studio' | 'builtin:<name>' | an install id) and writes the cookie. Classic,
 *   Studio and a builtin theme apply at once; an installed theme is stored first, then the page
 *   reloads into it (PA-13: its first paint comes from the server). A refused or failed PUT puts
 *   the tab and the cookie back on the style it had, unless a newer pick has replaced it.
 * - Checking (`ThemeChoiceSync`, mounted once inside the workspace and query providers): once per
 *   workspace and tab, `GET /api/themes/active`. A server choice that differs from this tab's
 *   style rewrites the cookie; an installed one reloads the page, at most once per tab (R16: the
 *   guard is in sessionStorage, and with no sessionStorage the tab never reloads). Safe mode asks
 *   nothing. The appearance-defaults step (ui-style-context.tsx) ran at mount, before any
 *   workspace was known, so a server choice always lands after it. An answer is dropped when a
 *   pick was made while it was in flight, or when the tab left the installed theme it was on
 *   (the host's three-failure revert to Studio, A-04c): the answer can only name the choice the
 *   server held before, and following it would reload straight back into the failing theme.
 * - The answer's `source` says where its choice came from: the member's stored choice, the
 *   workspace default, or Studio's fallback. With no stored choice (`default` or `fallback`), a
 *   tab on anything but Studio holds a pick only its cookie knows (made before PA-12): the first
 *   check PUTs that style as the member's choice instead of following the default, so a workspace
 *   default never overrides it. A tab on Studio, the build's own default, follows the default. A
 *   refused PUT (members may not choose, the install is gone) follows the server as below.
 * - Without `source` (a backend before it), 'studio' is both a stored Studio and "no choice, no
 *   default"; so a server Studio never moves a tab off Classic or a builtin theme (a pick made
 *   before PA-12; builtin Play until A-18 moves it). It does move a tab off an installed theme,
 *   which only the server can give. Any other server choice wins.
 * - The installs (`/api/themes/installed`) are asked for only once the theme menu, ⌘K or the
 *   Play note needs them (`wantInstalledThemes` / `useWantInstalledThemes`), never in safe mode;
 *   until then the menu shows what Settings › Appearance already fetched, if anything.
 * - The menu's entries (`themeChoiceEntries`): the builtin themes, then the workspace's installs
 *   as `theme:install:<id>`, "Private · not reviewed" when unreviewed. D1-c (Gerard, 18 Oct):
 *   once a builtin theme's own package (`automatos/<name>`) is installed, the builtin entry is
 *   hidden, except from a member still on it, who keeps it until A-18 moves them.
 */
import { createElement, useCallback, useContext, useEffect, useRef, useSyncExternalStore } from 'react'
import { defaultContext } from '@tanstack/react-query'
import { toast } from 'sonner'

import { PRIVATE_NOT_REVIEWED, isDisabled, themeName } from '@/components/settings/themes/theme-labels'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { persistUiStyle, useUiStyleOptional, type UiStyleContextValue } from '@/contexts/ui-style-context'
import { choiceOf, choiceSourceOf, themeQueryKeys, useInstalledThemes } from '@/hooks/use-installed-themes'
import { apiClient, type InstalledTheme } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'
import { BUILTIN_THEMES, isInstalledThemeStyle, themeRefOf, themeStyleOf } from '@/lib/theme-host/registry'
import type { ThemeRef, UiStyle } from '@/lib/theme-host/registry-types'
import { isSafeModeRequested, safeSessionStorage } from '@/lib/theme-host/safe-mode'
import { DEFAULT_UI_STYLE } from '@/lib/ui-style'

const LOG_PREFIX = '[theme-choice]'
const BUILTIN_REF_PREFIX = 'builtin:'
const INSTALL_REF_PREFIX = 'install:'
/** The publisher of the packages the builtin themes ship as (D1-c). */
export const FIRST_PARTY_PUBLISHER = 'automatos'
/** The note on a builtin entry its own installed package has replaced, shown only to a member still on it. */
export const BUILT_IN_NOTE = 'Built in'
export const CHOICE_NOT_SAVED = 'Your theme choice was not saved.'
const CORE_STYLES: ReadonlySet<string> = new Set(['classic', 'studio'])

/** This tab's record of the check, per workspace: checked, or checked and reloaded once. */
export const CHOICE_CHECK_KEY_PREFIX = 'automatos-theme-choice:'
const CHECKED = 'checked'
const RELOADED = 'reloaded'

// ============= CHOICE ↔ STYLE =============

/** The body of PUT /api/themes/choice for a style. */
export function choiceOfStyle(style: UiStyle): string {
  const ref = themeRefOf(style)
  if (ref === null) return style
  return ref.startsWith(INSTALL_REF_PREFIX) ? ref.slice(INSTALL_REF_PREFIX.length) : ref
}

/** The style a server choice names, or null for one this build does not know. */
export function styleOfChoice(choice: string | null): UiStyle | null {
  if (choice === null) return null
  if (CORE_STYLES.has(choice)) return choice as UiStyle
  if (choice.startsWith(BUILTIN_REF_PREFIX)) {
    const style = themeStyleOf(choice as ThemeRef)
    return BUILTIN_THEMES.some((entry) => themeStyleOf(entry.ref) === style) ? style : null
  }
  const installed = themeStyleOf(`${INSTALL_REF_PREFIX}${choice}`)
  return isInstalledThemeStyle(installed) ? installed : null
}

/**
 * True when the server holds no choice for this member (`source` default or fallback) and the tab
 * is on a style of its own that the answer does not name: the first check stores the tab's style.
 * Studio is the build's default, not a pick, so a tab on it follows the workspace default.
 */
export function storesTabStyle(current: UiStyle, active: unknown): boolean {
  const source = choiceSourceOf(active)
  if (source === null || source === 'choice' || current === DEFAULT_UI_STYLE) return false
  return styleOfChoice(choiceOf(active)) !== current
}

/**
 * The style this tab should move to after the server's answer, or null to stay. A server Studio
 * moves a tab only off an installed theme (one the server no longer gives this member, such as an
 * uninstalled one); off any other style it may only mean the server holds no choice (the header).
 */
export function reconcileTarget(current: UiStyle, active: unknown): UiStyle | null {
  const target = styleOfChoice(choiceOf(active))
  if (target === null || target === current) return null
  if (target === DEFAULT_UI_STYLE && !isInstalledThemeStyle(current)) return null
  return target
}

// ============= THE MENU'S ENTRIES =============

export interface ThemeChoiceEntry {
  readonly style: UiStyle
  readonly name: string
  readonly kind: 'core' | 'builtin' | 'installed'
  /** 'Private · not reviewed' for an unreviewed install, 'Built in' for a replaced builtin; else null. */
  readonly note: string | null
}

/** Classic and Studio, as ⌘K lists them before the themes (the menu draws its own, with icons). */
export const CORE_STYLE_CHOICES: readonly ThemeChoiceEntry[] = [
  { style: 'classic', name: 'Classic', kind: 'core', note: null },
  { style: 'studio', name: 'Studio', kind: 'core', note: null },
]

/** D1-c: the package a builtin theme ships as, `builtin:<name>` → `automatos/<name>`. */
export function packageIdOfBuiltin(ref: ThemeRef): string | null {
  return ref.startsWith(BUILTIN_REF_PREFIX) ? `${FIRST_PARTY_PUBLISHER}/${ref.slice(BUILTIN_REF_PREFIX.length)}` : null
}

function installedEntry(install: InstalledTheme): ThemeChoiceEntry | null {
  const style = themeStyleOf(`${INSTALL_REF_PREFIX}${install.installId}`)
  if (!isInstalledThemeStyle(style)) return null
  return { style, name: themeName(install), kind: 'installed', note: install.reviewed ? null : PRIVATE_NOT_REVIEWED }
}

/** The builtin entries, less those whose own package is installed (D1-c) unless `current` is on one. */
function builtinEntries(installedIds: ReadonlySet<string>, current: UiStyle | null): ThemeChoiceEntry[] {
  return BUILTIN_THEMES.flatMap((entry) => {
    const style = themeStyleOf(entry.ref)
    const replaced = installedIds.has(packageIdOfBuiltin(entry.ref) ?? '')
    if (replaced && style !== current) return []
    return [{ style, name: entry.name, kind: 'builtin' as const, note: replaced ? BUILT_IN_NOTE : null }]
  })
}

/**
 * The themes the menu and ⌘K offer, after Classic and Studio: the builtin ones, then the
 * workspace's installs. A disabled (withdrawn) install is neither offered nor replaces a builtin.
 */
export function themeChoiceEntries(installs: readonly InstalledTheme[], current: UiStyle | null): ThemeChoiceEntry[] {
  const usable = installs.filter((install) => !isDisabled(install))
  const installedIds = new Set(usable.map((install) => install.themeId))
  const installed = usable.map(installedEntry).filter((entry): entry is ThemeChoiceEntry => entry !== null)
  return [...builtinEntries(installedIds, current), ...installed]
}

/** Every style ⌘K can switch to: Classic, Studio, then `themeChoiceEntries`. */
export function styleChoices(installs: readonly InstalledTheme[], current: UiStyle | null): ThemeChoiceEntry[] {
  return [...CORE_STYLE_CHOICES, ...themeChoiceEntries(installs, current)]
}

// ============= THE TAB'S CHECK =============

/** Workspaces this document has checked (or is checking): one GET each, even with no sessionStorage. */
const checkedInThisDocument = new Set<string>()
/** Picks made in this document; a check answered after a pick is dropped (the pick is newer). */
let picksMade = 0

/** Whether the menu, ⌘K or the Play note has asked for the installs in this document. */
let installsWanted = false
const installsWantedListeners = new Set<() => void>()

/** Tests only: a new document (a reload) forgets what this one checked; sessionStorage keeps the tab's record. */
export function forgetChecksInThisDocumentForTests(): void {
  checkedInThisDocument.clear()
  picksMade = 0
  installsWanted = false
}

/**
 * The theme menu (on open), ⌘K (on open) and the Play note call this when they need the
 * workspace's installs: ThemeChoiceSync then fetches them, once, outside safe mode.
 */
export function wantInstalledThemes(): void {
  if (installsWanted) return
  installsWanted = true
  installsWantedListeners.forEach((listener) => listener())
}

/** `wantInstalledThemes` from a component, once `when` is true (a menu or dialog open). */
export function useWantInstalledThemes(when = true): void {
  useEffect(() => {
    if (when) wantInstalledThemes()
  }, [when])
}

function subscribeInstallsWanted(listener: () => void): () => void {
  installsWantedListeners.add(listener)
  return () => {
    installsWantedListeners.delete(listener)
  }
}

const installsWantedNow = (): boolean => installsWanted
const installsWantedOnServer = (): boolean => false

function inSafeMode(): boolean {
  return typeof window !== 'undefined' && isSafeModeRequested(window.location.search, safeSessionStorage())
}

function checkKey(workspaceId: string): string {
  return `${CHOICE_CHECK_KEY_PREFIX}${workspaceId}`
}

function checkedInThisTab(workspaceId: string): boolean {
  if (checkedInThisDocument.has(workspaceId)) return true
  try {
    return safeSessionStorage()?.getItem(checkKey(workspaceId)) != null
  } catch (error) {
    console.warn(LOG_PREFIX, 'the tab could not read its theme check', error)
    return false
  }
}

/** Records the check for this tab; true only when sessionStorage kept it (the reload guard, R16). */
function recordCheck(workspaceId: string, value: string): boolean {
  checkedInThisDocument.add(workspaceId)
  const storage = safeSessionStorage()
  if (storage === null) return false
  try {
    storage.setItem(checkKey(workspaceId), value)
    return storage.getItem(checkKey(workspaceId)) === value
  } catch (error) {
    console.warn(LOG_PREFIX, 'the tab could not record its theme check', error)
    return false
  }
}

/** Moves this tab to the server's choice: an installed one reloads once, or waits for the next page. */
function applyServerStyle(ui: UiStyleContextValue, workspaceId: string, target: UiStyle): void {
  if (!isInstalledThemeStyle(target)) {
    recordCheck(workspaceId, CHECKED)
    ui.setStyle(target)
    return
  }
  if (recordCheck(workspaceId, RELOADED)) ui.setStyle(target)
  else persistUiStyle(target)
}

/**
 * Whether the tab moved on while the check was in flight: a pick, or the tab leaving the installed
 * theme it was on when the check was sent (the host's revert, which is no pick).
 */
function movedOnSince(picksBefore: number, styleBefore: UiStyle | null, current: UiStyleContextValue): boolean {
  if (picksMade !== picksBefore) return true
  return styleBefore !== null && isInstalledThemeStyle(styleBefore) && current.style !== styleBefore
}

/** PUTs the tab's cookie-only style as the member's choice; true when the server stored it. */
async function storeTabStyle(style: UiStyle): Promise<boolean> {
  try {
    await apiClient.putThemeChoice(choiceOfStyle(style))
    return true
  } catch (error) {
    console.warn(LOG_PREFIX, "the tab's theme could not be stored as the member's choice", error)
    return false
  }
}

/** Once per workspace and tab, outside safe mode: GET /api/themes/active, then store the tab's own style or follow the server's. */
function useServerChoiceCheck(workspaceId: string): void {
  const ui = useUiStyleOptional()
  const uiRef = useRef(ui)
  uiRef.current = ui
  const queryClient = useContext(defaultContext)

  useEffect(() => {
    if (checkedInThisTab(workspaceId)) return
    if (inSafeMode()) return
    checkedInThisDocument.add(workspaceId)
    const picksBefore = picksMade
    const styleBefore = uiRef.current?.style ?? null
    const stillCurrent = (): UiStyleContextValue | null => {
      const current = uiRef.current
      return current === null || movedOnSince(picksBefore, styleBefore, current) ? null : current
    }
    const settle = async (active: unknown): Promise<void> => {
      const current = stillCurrent()
      if (current === null) return
      if (storesTabStyle(current.style, active) && (await storeTabStyle(current.style))) {
        recordCheck(workspaceId, CHECKED)
        void queryClient?.invalidateQueries({ queryKey: themeQueryKeys.active(workspaceId) })
        return
      }
      const now = stillCurrent()
      if (now === null) return
      const target = reconcileTarget(now.style, active)
      if (target === null) recordCheck(workspaceId, CHECKED)
      else applyServerStyle(now, workspaceId, target)
    }
    apiClient
      .getActiveTheme()
      .then(settle)
      .catch((error: unknown) => console.warn(LOG_PREFIX, 'the theme choice could not be checked', error))
  }, [workspaceId, queryClient])
}

/**
 * Hands the workspace's installs to the style context, where the menu and ⌘K read them. They are
 * fetched only once wanted (wantInstalledThemes) and never in safe mode; until then this reads the
 * cache only, which Settings › Appearance fills.
 */
function usePublishInstalledThemes(): void {
  const setInstalledThemes = useUiStyleOptional()?.setInstalledThemes
  const wanted = useSyncExternalStore(subscribeInstallsWanted, installsWantedNow, installsWantedOnServer)
  const { data } = useInstalledThemes({ enabled: wanted && !inSafeMode() })
  useEffect(() => {
    if (setInstalledThemes) setInstalledThemes(data ?? [])
  }, [data, setInstalledThemes])
}

function ThemeChoiceSyncInWorkspace({ workspaceId }: { workspaceId: string }): null {
  usePublishInstalledThemes()
  useServerChoiceCheck(workspaceId)
  return null
}

/**
 * Mount once, inside the workspace and query providers (components/shared/global-search.tsx).
 * Renders nothing; outside a workspace, or with no query client (isolated mounts), it does nothing.
 */
export function ThemeChoiceSync() {
  const hasQueryClient = useContext(defaultContext) !== undefined
  const workspaceId = useWorkspaceOptional()?.workspace?.id ?? null
  if (!hasQueryClient || workspaceId === null) return null
  return createElement(ThemeChoiceSyncInWorkspace, { workspaceId })
}

// ============= CHOOSING =============

interface StoreChoice {
  readonly next: UiStyle
  /** The style the tab had before this pick: put back when the server does not store the pick. */
  readonly previous: UiStyle
  /** This pick's number in `picksMade`: a newer pick is never undone by this one's failure. */
  readonly pick: number
  readonly workspaceId: string
  readonly setStyle: (style: UiStyle) => void
  readonly refresh: () => void
}

/**
 * PUT the choice; Classic, Studio and a builtin theme are already showing, an installed one reloads
 * after. When the PUT fails, a pick already showing is undone (style and cookie), so the tab never
 * keeps a choice the server does not hold.
 */
async function storeChoice({ next, previous, pick, workspaceId, setStyle, refresh }: StoreChoice): Promise<void> {
  const installed = isInstalledThemeStyle(next)
  if (!installed) setStyle(next)
  try {
    await apiClient.putThemeChoice(choiceOfStyle(next))
  } catch (error) {
    toast.error(CHOICE_NOT_SAVED, { description: themeErrorMessage(error) })
    if (!installed && picksMade === pick) setStyle(previous)
    return
  }
  recordCheck(workspaceId, CHECKED)
  refresh()
  if (installed) setStyle(next)
}

/**
 * The theme menu's and ⌘K's pick: stored on the server in a workspace, then applied (above).
 * Outside a workspace (isolated mounts) it only applies the style, as before PA-12.
 */
export function useChooseStyle(): (next: UiStyle) => void {
  const ui = useUiStyleOptional()
  const workspaceId = useWorkspaceOptional()?.workspace?.id ?? null
  const queryClient = useContext(defaultContext)

  return useCallback(
    (next: UiStyle) => {
      if (ui === null || next === ui.style) return
      picksMade += 1
      if (workspaceId === null) {
        ui.setStyle(next)
        return
      }
      const refresh = () => void queryClient?.invalidateQueries({ queryKey: themeQueryKeys.active(workspaceId) })
      void storeChoice({ next, previous: ui.style, pick: picksMade, workspaceId, setStyle: ui.setStyle, refresh })
    },
    [ui, workspaceId, queryClient],
  )
}
