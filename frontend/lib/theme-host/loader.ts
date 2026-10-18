/**
 * PRD-261 A-01, A-04a — the steps every theme load shares, in the order PRD-261 §4.3 item 4 sets:
 * check the manifest's SDK range (a mismatch is refused and the entry is never requested),
 * import the entry, call register(host), then apply the stylesheet and preset. Where a theme comes
 * from (the active install, or a module compiled into the app) is sources.ts.
 *
 * A load never throws to its caller: settleLoad turns every failure into a result with its
 * reason, logged, so the provider can fall back to Studio.
 */
import { FRAME_SLOT, THEME_HOST_VERSION, THEME_PAGE_ROUTES, THEME_ROUTE_PATTERN, THEME_SLOT_IDS } from './constants'
import { satisfiesSdkRange } from './engines'
import { applyLook } from './look'
import { parseLookPreset } from './manifest'
import type { SlotId, ThemeRef } from './registry-types'
import type {
  FrameComponent,
  LoaderDeps,
  SpikeManifest,
  ThemeDefinition,
  ThemeFailureReason,
  ThemeLoadOk,
  ThemeLoadResult,
  ThemeRegistrationHost,
  ThemeRegistrations,
} from './types'

export type { LoaderDeps } from './types'

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

/** Runs one step; whatever it throws becomes a failure with `reason`, unless it already has one. */
export async function step<T>(reason: ThemeFailureReason, run: () => T | Promise<T>): Promise<T> {
  try {
    return await run()
  } catch (error) {
    throw error instanceof ThemeStepError ? error : new ThemeStepError(reason, error)
  }
}

/** Every request and import a load makes is bounded, so a stalled one becomes a failure, not a hang. */
export function withTimeout<T>(pending: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs)
  })
  return Promise.race([pending, timeout]).finally(() => clearTimeout(timer))
}

function isThemeDefinition(value: unknown): value is ThemeDefinition {
  return typeof value === 'object' && value !== null && typeof (value as { register?: unknown }).register === 'function'
}

/** The theme definition a loaded module exports as its default, or a throw saying what is missing. */
export function definitionOf(loaded: unknown): ThemeDefinition {
  const definition = (loaded as { default?: unknown } | null | undefined)?.default
  if (!isThemeDefinition(definition)) throw new Error('the theme entry has no default export with a register(host) function')
  return definition
}

/** A component React can render: a function (or class), or an exotic one such as lazy, memo or forwardRef. */
function isComponent(value: unknown): boolean {
  if (typeof value === 'function') return true
  return typeof value === 'object' && value !== null && '$$typeof' in value
}

interface Collected {
  readonly slots: Map<SlotId, unknown>
  readonly pages: Map<string, unknown>
  readonly routes: Map<string, unknown>
}

/** Keeps one registration, refusing an unknown key, a non-component or a second one for the same key. */
function keep<K extends string>(into: Map<K, unknown>, key: K, component: unknown, allowed: (key: string) => boolean, what: string): void {
  if (!allowed(key)) throw new Error(`${what} "${String(key)}" is not one this host serves`)
  if (!isComponent(component)) throw new Error(`${what} "${key}" needs a component`)
  if (into.has(key)) throw new Error(`${what} "${key}" was registered twice`)
  into.set(key, component)
}

function collectingHost(sdkVersion: string, collected: Collected): ThemeRegistrationHost {
  return Object.freeze({
    sdkVersion,
    hostVersion: THEME_HOST_VERSION,
    registerSlot(slot: SlotId, component: unknown): void {
      keep(collected.slots, slot, component, (key) => (THEME_SLOT_IDS as readonly string[]).includes(key), 'slot')
    },
    registerPage(route: string, component: unknown): void {
      keep(collected.pages, route, component, (key) => THEME_PAGE_ROUTES.includes(key), 'page')
    },
    registerRoute(path: string, component: unknown): void {
      keep(collected.routes, path, component, (key) => THEME_ROUTE_PATTERN.test(key), 'route')
    },
  })
}

/**
 * Runs register(host) and returns what it registered; a frame is required. A register that
 * throws, or registers no frame, is unregistered before the failure goes on, so whatever it set
 * up before failing does not outlive the load.
 */
export function registerTheme(definition: ThemeDefinition, sdkVersion: string): ThemeRegistrations {
  const collected: Collected = { slots: new Map(), pages: new Map(), routes: new Map() }
  try {
    definition.register(collectingHost(sdkVersion, collected))
    const frame = collected.slots.get(FRAME_SLOT)
    if (frame === undefined) throw new Error(`the theme registered no "${FRAME_SLOT}" component`)
    return { frame: frame as FrameComponent, slots: collected.slots, pages: collected.pages, routes: collected.routes }
  } catch (error) {
    runLogged(() => definition.unregister?.())
    throw error
  }
}

function checkEngines(manifest: SpikeManifest, sdkVersion: string): void {
  if (!satisfiesSdkRange(manifest.sdkRange, sdkVersion)) {
    throw new Error(`engines.sdk "${manifest.sdkRange}" does not accept the host's SDK ${sdkVersion}`)
  }
}

async function importEntry(manifest: SpikeManifest, deps: LoaderDeps): Promise<ThemeDefinition> {
  const timedOut = `the theme entry did not load within ${deps.timeoutMs} ms`
  return definitionOf(await withTimeout(deps.importModule(manifest.entry), deps.timeoutMs, timedOut))
}

async function applyPresetLook(manifest: SpikeManifest, deps: LoaderDeps): Promise<() => void> {
  const timedOut = `GET ${manifest.preset} did not answer within ${deps.timeoutMs} ms`
  const parsed = parseLookPreset(await withTimeout(deps.fetchJson(manifest.preset), deps.timeoutMs, timedOut))
  if (!parsed.ok) throw new Error(parsed.detail)
  return applyLook(manifest, parsed.preset, deps.doc, deps.timeoutMs)
}

export function runLogged(task: () => void): void {
  try {
    task()
  } catch (error) {
    console.error(LOG_PREFIX, 'cleanup', error)
  }
}

/** The ok result's cleanup: the look's, then the theme's unregister, once. */
export function cleanupOnce(lookCleanup: () => void, definition: ThemeDefinition): () => void {
  let done = false
  return () => {
    if (done) return
    done = true
    runLogged(lookCleanup)
    runLogged(() => definition.unregister?.())
  }
}

/** The steps after a checked manifest: engines, entry, register, look. */
export async function loadFromManifest(
  ref: ThemeRef,
  manifest: SpikeManifest,
  sdkVersion: string,
  deps: LoaderDeps,
): Promise<ThemeLoadOk> {
  await step('engines', () => checkEngines(manifest, sdkVersion))
  const definition = await step('load', () => importEntry(manifest, deps))
  const registrations = await step('register', () => registerTheme(definition, sdkVersion))
  let lookCleanup: () => void
  try {
    lookCleanup = await step('look', () => applyPresetLook(manifest, deps))
  } catch (error) {
    runLogged(() => definition.unregister?.())
    throw error
  }
  return { ok: true, ref, manifest, ...registrations, cleanup: cleanupOnce(lookCleanup, definition) }
}

/** Runs a load to its result. Never throws: a failure is logged and returned with its reason. */
export async function settleLoad(run: () => Promise<ThemeLoadOk>): Promise<ThemeLoadResult> {
  try {
    return await run()
  } catch (error) {
    const reason: ThemeFailureReason = error instanceof ThemeStepError ? error.reason : 'load'
    const original = error instanceof ThemeStepError ? error.original : error
    console.error(LOG_PREFIX, reason, original)
    return { ok: false, reason, detail: error instanceof Error ? error.message : String(error) }
  }
}
