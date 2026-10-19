/**
 * PRD-261 A-10b — SDK area `theme` (frontend/lib/theme-sdk/theme.ts): the SDK's version and defineTheme.
 */
import type { SlotId } from '../internal/host'

/**
 * What register(host) receives: the two versions and the three ways to hand the host a component.
 */
export interface ThemeRegistrationHost {
  readonly sdkVersion: string
  readonly hostVersion: string
  registerSlot(slot: SlotId, component: unknown): void
  /** Only the routes of the host's theme page list (the Q3 pages). */
  registerPage(route: string, component: unknown): void
  /** Served at /x/<path>. */
  registerRoute(path: string, component: unknown): void
}

/** A theme's default export, as defineTheme returns it. */
export interface ThemeDefinition {
  register(host: ThemeRegistrationHost): void
  unregister?(): void
}

/** The version of the SDK the host serves; this package's version. */
export declare const SDK_VERSION: string

/** A theme's definition, checked and frozen: it must carry a register(host) function, or this throws a TypeError. */
export declare function defineTheme(definition: ThemeDefinition): Readonly<ThemeDefinition>
