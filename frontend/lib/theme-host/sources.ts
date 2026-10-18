/**
 * PRD-261 A-04a — where a theme comes from (plan §2.2 `ThemeSource`).
 *
 * - **installed** (`install:<uuid>`): reads GET /api/themes/active through the backend client,
 *   checks that it names this install and carries a sha256 digest, checks its manifest with every
 *   file under `/api/themes/files/<digest>/`, then imports the entry with the browser's own
 *   import(), so its bare imports resolve through the page's import map to the host's modules.
 * - **builtin** (`builtin:<id>`): a module compiled into the app, loaded with a webpack import();
 *   its frame registers synchronously as soon as that import resolves. Builtin Play keeps its
 *   compiled path until A-04b registers it here.
 *
 * The active-theme request goes through apiClient's named method (getActiveTheme), so the
 * route-contract check sees it; a 404 from it means the workspace has no active install (a stale
 * `theme:install:` cookie), which is no theme to load, not a failure. The preset fetch below is a
 * same-origin static file under the theme's files folder, not the backend client, so it is a
 * plain fetch.
 */
import { apiClient } from '@/lib/api-client'
import type { UiStyle } from '@/lib/ui-style'

import { THEME_ACTIVE_ENDPOINT, THEME_LOAD_TIMEOUT_MS } from './constants'
import { exposeHostModules } from './host-modules'
import { cleanupOnce, definitionOf, loadFromManifest, registerTheme, settleLoad, step, withTimeout } from './loader'
import { parseActiveInstall, parseSpikeManifest } from './manifest'
import { importThemeModule } from './native-import'
import { isInstalledThemeStyle, themeRefOf } from './registry'
import type { ThemeRef } from './registry-types'
import { SDK_VERSION } from './sdk'
import type { LoaderDeps, SpikeManifest, ThemeLoadAbsent, ThemeLoadOk, ThemeLoadResult, ThemeSource } from './types'

const INSTALL_REF_PREFIX = 'install:'
/** What GET /api/themes/active answers when the workspace has no active install. */
const NO_ACTIVE_INSTALL_STATUS = 404

/** The HTTP status apiClient puts on the error it throws, if it has one. */
function httpStatusOf(error: unknown): unknown {
  return (error as { status?: unknown } | null)?.status
}

/** GET /api/themes/active through apiClient; null when it answers 404: no active install. */
export async function fetchActiveInstall(): Promise<unknown> {
  try {
    return await apiClient.getActiveTheme()
  } catch (error) {
    if (httpStatusOf(error) === NO_ACTIVE_INSTALL_STATUS) return null
    throw error
  }
}

function noActiveInstall(): ThemeLoadAbsent {
  return { ok: false, reason: null, detail: `GET ${THEME_ACTIVE_ENDPOINT} answered ${NO_ACTIVE_INSTALL_STATUS}: no active install` }
}

async function fetchSameOriginJson(url: string): Promise<unknown> {
  const response = await fetch(url, { credentials: 'same-origin', cache: 'no-store' })
  if (!response.ok) throw new Error(`GET ${url} answered ${response.status}`)
  return response.json()
}

export function defaultLoaderDeps(): LoaderDeps {
  return {
    exposeHost: exposeHostModules,
    fetchActive: fetchActiveInstall,
    fetchJson: fetchSameOriginJson,
    importModule: importThemeModule,
    doc: document,
    timeoutMs: THEME_LOAD_TIMEOUT_MS,
  }
}

/** The active install's checked manifest, with its files resolved under the package's folder; null with no active install. */
async function readActiveManifest(installId: string, deps: LoaderDeps): Promise<SpikeManifest | null> {
  const timedOut = `GET ${THEME_ACTIVE_ENDPOINT} did not answer within ${deps.timeoutMs} ms`
  const answer = await withTimeout(deps.fetchActive(), deps.timeoutMs, timedOut)
  if (answer === null) return null
  const active = parseActiveInstall(answer, installId)
  if (!active.ok) throw new Error(active.detail)
  const parsed = parseSpikeManifest(active.manifest, active.filesRoot)
  if (!parsed.ok) throw new Error(parsed.detail)
  return parsed.manifest
}

/** The theme installed in the workspace as `installId` (a lowercase uuid). */
export function installedThemeSource(installId: string): ThemeSource {
  const ref: ThemeRef = `${INSTALL_REF_PREFIX}${installId}`
  return {
    ref,
    load: (deps) =>
      settleLoad(async (): Promise<ThemeLoadOk | ThemeLoadAbsent> => {
        const host = await step('load', () => deps.exposeHost())
        const manifest = await step('manifest', () => readActiveManifest(installId, deps))
        if (manifest === null) return noActiveInstall()
        return loadFromManifest(ref, manifest, host.sdkVersion, deps)
      }),
  }
}

/**
 * A theme compiled into the app. `importDefinition` is a webpack `import()` of its module (a
 * default export made with defineTheme). It renders with the app's own React, so the host's
 * module global is not installed, and its look comes with its chunk and the boot data (A-03).
 */
export function builtinThemeSource(ref: `builtin:${string}`, importDefinition: () => Promise<unknown>): ThemeSource {
  return {
    ref,
    load: (deps) =>
      settleLoad(async (): Promise<ThemeLoadOk> => {
        const timedOut = `the builtin theme ${ref} did not load within ${deps.timeoutMs} ms`
        const definition = await step('load', async () => definitionOf(await withTimeout(importDefinition(), deps.timeoutMs, timedOut)))
        const registrations = await step('register', () => registerTheme(definition, SDK_VERSION))
        return { ok: true, ref, manifest: null, ...registrations, cleanup: cleanupOnce(() => undefined, definition) }
      }),
  }
}

/** The install a style names, or null for Classic, Studio and a builtin theme. */
export function installIdOf(style: UiStyle): string | null {
  if (!isInstalledThemeStyle(style)) return null
  const ref = themeRefOf(style)
  return ref === null ? null : ref.slice(INSTALL_REF_PREFIX.length)
}

/** The source the host loads for a style: only an installed theme in A-04a (builtin Play: A-04b). */
export function sourceForStyle(style: UiStyle): ThemeSource | null {
  const installId = installIdOf(style)
  return installId === null ? null : installedThemeSource(installId)
}

/** Loads the theme a style names. Never throws; a style with no source is a 'load' failure. */
export async function loadThemeForStyle(style: UiStyle, deps: LoaderDeps = defaultLoaderDeps()): Promise<ThemeLoadResult> {
  const source = sourceForStyle(style)
  if (source === null) return { ok: false, reason: 'load', detail: `no theme source for the style ${style}` }
  return source.load(deps)
}
