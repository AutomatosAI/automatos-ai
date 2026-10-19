/**
 * PRD-261 A-01, A-04a — the theme host's types: the global a theme's shims read, what a theme
 * registers with the host, a theme's checked manifest and preset, where a theme comes from (its
 * source) and what a load returns.
 */
import type { ComponentType, ReactNode } from 'react'

import type { SlotId, ThemeRef } from './registry-types'

/** The frozen global the host puts its shared modules on (lib/theme-host/host-modules.ts). */
export interface ThemeHostGlobal {
  readonly sdkVersion: string
  /** Each shared specifier (shared-modules.json) and the host's own instance of it. */
  readonly modules: Readonly<Record<string, unknown>>
}

/**
 * What register(host) receives: the two versions and the three ways to hand the host a
 * component. Pages and routes are accepted and kept; A-04b renders them.
 */
export interface ThemeRegistrationHost {
  readonly sdkVersion: string
  readonly hostVersion: string
  registerSlot(slot: SlotId, component: unknown): void
  /** Only the routes of THEME_PAGE_ROUTES (the Q3 page list). */
  registerPage(route: string, component: unknown): void
  /** Served at /x/<path> (A-04b). */
  registerRoute(path: string, component: unknown): void
}

/** A theme's default export, as defineTheme returns it. */
export interface ThemeDefinition {
  register(host: ThemeRegistrationHost): void
  unregister?(): void
}

/**
 * Why a theme did not start, or stopped: each reason has its own notice text.
 *
 * - `auth` (A-08b): the session was refused (401/403) on the active-theme read, the entry or a
 *   theme file, as when it expired. It is the session's fault, not the theme's, so it never
 *   counts toward the three-failure revert (A-04c).
 * - `revoked`: the installed version was revoked. A `warn` revocation never fails the load (the
 *   theme runs and the host shows a notice, ThemeLoadOk.revoked); a `disable` one names no theme
 *   (ThemeLoadAbsent), as the server's own choice rewrite does.
 * - `opt-in`: private code the member has not opted in to (PRD-261 Q6): GET /api/themes/active
 *   says `optIn.required`, so no theme file is asked for (the files proxy would 404 each one).
 * - `unavailable`: the theme's files could not be reached (a 5xx from the files proxy, a network
 *   failure, or the host's own sources chunk not loading, as after a deploy). Not the theme's
 *   fault, so it never counts toward the revert either.
 */
export type ThemeFailureReason = 'manifest' | 'engines' | 'load' | 'register' | 'look' | 'render'
  | 'auth' | 'revoked' | 'opt-in' | 'unavailable'

/** What the host is doing: no theme here, loading one, showing one, fell back, or safe mode. */
export type ThemeHostStatus = 'off' | 'loading' | 'active' | 'failed' | 'safe-mode'

/** The props of slot 'frame@1': the page, and the two things MainLayout hands every chrome. */
export interface FrameProps {
  readonly children?: ReactNode
  readonly fullBleed?: boolean
  readonly autoWidget?: ReactNode
}

/** The component a theme registers for the frame slot; it draws its frame around `children`. */
export type FrameComponent = ComponentType<FrameProps>

/**
 * A checked theme.json: a package's (`automatos-theme/v1`, its files by the package layout) or the
 * browser lane's spike fixtures' (`automatos-theme-spike/v1`, its files named), manifest.ts
 * parseThemeManifest. The three file fields are absolute URLs under the theme's files folder.
 */
export interface SpikeManifest {
  readonly id: string
  readonly name: string
  readonly version: string
  readonly sdkRange: string
  readonly htmlClass: string
  readonly slots: readonly string[]
  readonly entry: string
  readonly styleSheet: string
  readonly preset: string
}

/** A checked look preset: CSS custom properties and their values. */
export interface LookPreset {
  readonly vars: Readonly<Record<string, string>>
}

/** Everything a theme's register(host) handed over. The frame is required; the rest may be empty. */
export interface ThemeRegistrations {
  readonly frame: FrameComponent
  readonly slots: ReadonlyMap<SlotId, unknown>
  readonly pages: ReadonlyMap<string, unknown>
  readonly routes: ReadonlyMap<string, unknown>
}

/** A `warn` revocation of the running version (GET /api/themes/active `theme.revoked`): it still runs. */
export interface ThemeRevocationWarning {
  readonly level: 'warn'
  /** The maintainer's reason, shown as text; empty when none was given. */
  readonly reason: string
}

export interface ThemeLoadOk extends ThemeRegistrations {
  readonly ok: true
  readonly ref: ThemeRef
  /** The installed theme's checked manifest; null for a builtin theme, which has a registry entry instead. */
  readonly manifest: SpikeManifest | null
  /** A `warn` revocation of this installed version, for the host's notice; absent or null for none. */
  readonly revoked?: ThemeRevocationWarning | null
  /** Removes the theme's class, stylesheet and variables, then runs its unregister; safe to call twice. */
  readonly cleanup: () => void
}

export interface ThemeLoadFailure {
  readonly ok: false
  readonly reason: ThemeFailureReason
  readonly detail: string
}

/**
 * No theme to load: GET /api/themes/active names no theme for the install (`theme: null`, a
 * `disable` revocation, another install, or a 404 from a backend without A-07b's route), as for a
 * stale `theme:install:` cookie. Not a failure: the host
 * turns off and Studio shows, with no notice. `reason` is null so it never reads as a failure's.
 */
export interface ThemeLoadAbsent {
  readonly ok: false
  readonly reason: null
  readonly detail: string
}

export type ThemeLoadResult = ThemeLoadOk | ThemeLoadFailure | ThemeLoadAbsent

/** Everything a load reaches outside itself, injected so tests can replace each one. */
export interface LoaderDeps {
  readonly exposeHost: () => ThemeHostGlobal
  /** GET /api/themes/active through the backend client (A-07b's route); null when it answers 404 (no active install). */
  readonly fetchActive: () => Promise<unknown>
  /** A same-origin JSON file under the theme's files folder (its preset). */
  readonly fetchJson: (url: string) => Promise<unknown>
  /** The browser's own import(), for an installed theme's entry. */
  readonly importModule: (url: string) => Promise<unknown>
  /** A-08c: leaves the package digest the active-theme answer named for the next request's boot cache. */
  readonly rememberDigest?: (installId: string, digest: string) => void
  readonly doc: Document
  readonly timeoutMs: number
}

/**
 * A-08c: the cookie in which the client host leaves the package digest GET /api/themes/active
 * named for an install, as `<installId>.<digest>`. The root layout's boot cache
 * (boot-server.ts) is keyed by user, install and this digest, so a boot whose digest the client
 * no longer sees active (after an update or a rollback) is fetched again instead of served for
 * the rest of its 60 s. A hint only: it can make the server ask the backend again, never choose
 * what the page paints.
 */
export const THEME_DIGEST_COOKIE = 'automatos-theme-digest'
/** Between the install id and the digest in THEME_DIGEST_COOKIE (neither contains it). */
export const THEME_DIGEST_HINT_SEPARATOR = '.'

/** Where a theme comes from: compiled into the app (`builtin:`) or installed in the workspace (`install:`). */
export interface ThemeSource {
  readonly ref: ThemeRef
  load(deps: LoaderDeps): Promise<ThemeLoadResult>
}

declare global {
  interface Window {
    __AUTOMATOS_THEME_HOST__?: ThemeHostGlobal
  }
}
