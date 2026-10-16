/**
 * PRD-261 A-01 — loads the spike's one theme, in the order PRD-261 §4.3 item 4 sets: share the
 * host's modules, read and check theme.json, check its SDK range (a mismatch is refused and the
 * entry is never requested), import the entry with the browser's own import(), call
 * register(host), then apply the stylesheet and preset.
 *
 * It never throws: every failure is logged with its reason and returned as a result, so the
 * provider can fall back to Studio. The manifest URL is a constant, and the files it names must
 * resolve under the theme's fixed folder (manifest.ts). This fetches fixed same-origin static
 * files, not the backend, so it does not go through apiClient.
 */
import { FRAME_SLOT, THEME_LOAD_TIMEOUT_MS, THEME_SPIKE_MANIFEST_PATH } from './constants'
import { satisfiesSdkRange } from './engines'
import { exposeHostModules } from './host-modules'
import { applyLook } from './look'
import { parseLookPreset, parseSpikeManifest } from './manifest'
import { importThemeModule } from './native-import'
import type {
  FrameComponent,
  SpikeManifest,
  ThemeDefinition,
  ThemeFailureReason,
  ThemeHostGlobal,
  ThemeLoadOk,
  ThemeLoadResult,
  ThemeRegistrationHost,
} from './types'

export interface LoaderDeps {
  readonly exposeHost: () => ThemeHostGlobal
  readonly fetchJson: (url: string) => Promise<unknown>
  readonly importModule: (url: string) => Promise<unknown>
  readonly doc: Document
  readonly timeoutMs: number
}

const LOG_PREFIX = '[theme-host]'

/** A failure at one step of the load, carrying the reason the provider shows. */
class ThemeStepError extends Error {
  readonly reason: ThemeFailureReason
  readonly original: unknown

  constructor(reason: ThemeFailureReason, original: unknown) {
    super(original instanceof Error ? original.message : String(original))
    this.name = 'ThemeStepError'
    this.reason = reason
    this.original = original
  }
}

async function step<T>(reason: ThemeFailureReason, run: () => T | Promise<T>): Promise<T> {
  try {
    return await run()
  } catch (error) {
    throw error instanceof ThemeStepError ? error : new ThemeStepError(reason, error)
  }
}

async function fetchSameOriginJson(url: string): Promise<unknown> {
  const response = await fetch(url, { credentials: 'same-origin', cache: 'no-store' })
  if (!response.ok) throw new Error(`GET ${url} answered ${response.status}`)
  return response.json()
}

export function defaultLoaderDeps(): LoaderDeps {
  return {
    exposeHost: exposeHostModules,
    fetchJson: fetchSameOriginJson,
    importModule: importThemeModule,
    doc: document,
    timeoutMs: THEME_LOAD_TIMEOUT_MS,
  }
}

function withTimeout<T>(pending: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs)
  })
  return Promise.race([pending, timeout]).finally(() => clearTimeout(timer))
}

/** Every request the load makes is bounded, so a stalled one becomes a failure, not a hang. */
function fetchJsonWithin(deps: LoaderDeps, url: string): Promise<unknown> {
  return withTimeout(deps.fetchJson(url), deps.timeoutMs, `GET ${url} did not answer within ${deps.timeoutMs} ms`)
}

function isThemeDefinition(value: unknown): value is ThemeDefinition {
  return typeof value === 'object' && value !== null && typeof (value as { register?: unknown }).register === 'function'
}

async function readManifest(deps: LoaderDeps): Promise<SpikeManifest> {
  const parsed = parseSpikeManifest(await fetchJsonWithin(deps, THEME_SPIKE_MANIFEST_PATH))
  if (!parsed.ok) throw new Error(parsed.detail)
  return parsed.manifest
}

function checkEngines(manifest: SpikeManifest, sdkVersion: string): void {
  if (!satisfiesSdkRange(manifest.sdkRange, sdkVersion)) {
    throw new Error(`engines.sdk "${manifest.sdkRange}" does not accept the host's SDK ${sdkVersion}`)
  }
}

async function importDefinition(manifest: SpikeManifest, deps: LoaderDeps): Promise<ThemeDefinition> {
  const timedOut = `the theme entry did not load within ${deps.timeoutMs} ms`
  const loaded = await withTimeout(deps.importModule(manifest.entry), deps.timeoutMs, timedOut)
  const definition = (loaded as { default?: unknown } | null | undefined)?.default
  if (!isThemeDefinition(definition)) throw new Error('the theme entry has no default export with a register(host) function')
  return definition
}

/** Runs register(host) and returns the frame it registered; the host accepts only the frame slot. */
function registerFrame(definition: ThemeDefinition, sdkVersion: string): FrameComponent {
  const slots = new Map<string, FrameComponent>()
  const host: ThemeRegistrationHost = Object.freeze({
    sdkVersion,
    registerSlot(slot: string, component: unknown): void {
      if (slot !== FRAME_SLOT || typeof component !== 'function') {
        throw new Error(`registerSlot accepts only "${FRAME_SLOT}" with a component function`)
      }
      slots.set(slot, component as FrameComponent)
    },
  })
  definition.register(host)
  const frame = slots.get(FRAME_SLOT)
  if (frame === undefined) throw new Error(`the theme registered no "${FRAME_SLOT}" component`)
  return frame
}

async function applyPresetLook(manifest: SpikeManifest, deps: LoaderDeps): Promise<() => void> {
  const parsed = parseLookPreset(await fetchJsonWithin(deps, manifest.preset))
  if (!parsed.ok) throw new Error(parsed.detail)
  return applyLook(manifest, parsed.preset, deps.doc, deps.timeoutMs)
}

function runLogged(task: () => void): void {
  try {
    task()
  } catch (error) {
    console.error(LOG_PREFIX, 'cleanup', error)
  }
}

/** The ok result's cleanup: the look's, then the theme's unregister, once. */
function cleanupOnce(lookCleanup: () => void, definition: ThemeDefinition): () => void {
  let done = false
  return () => {
    if (done) return
    done = true
    runLogged(lookCleanup)
    runLogged(() => definition.unregister?.())
  }
}

async function runLoad(deps: LoaderDeps): Promise<ThemeLoadOk> {
  const host = await step('load', () => deps.exposeHost())
  const manifest = await step('manifest', () => readManifest(deps))
  await step('engines', () => checkEngines(manifest, host.sdkVersion))
  const definition = await step('load', () => importDefinition(manifest, deps))
  const frame = await step('register', () => registerFrame(definition, host.sdkVersion))
  let lookCleanup: () => void
  try {
    lookCleanup = await step('look', () => applyPresetLook(manifest, deps))
  } catch (error) {
    runLogged(() => definition.unregister?.())
    throw error
  }
  return { ok: true, manifest, frame, cleanup: cleanupOnce(lookCleanup, definition) }
}

/** Loads the spike's theme. Never throws: a failure is logged and returned with its reason. */
export async function loadSpikeTheme(deps: LoaderDeps = defaultLoaderDeps()): Promise<ThemeLoadResult> {
  try {
    return await runLoad(deps)
  } catch (error) {
    const reason: ThemeFailureReason = error instanceof ThemeStepError ? error.reason : 'load'
    const original = error instanceof ThemeStepError ? error.original : error
    console.error(LOG_PREFIX, reason, original)
    return { ok: false, reason, detail: error instanceof Error ? error.message : String(error) }
  }
}
