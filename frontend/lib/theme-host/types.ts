/**
 * PRD-261 A-01 — the theme host's types: the global a theme's shims read, and what a theme
 * registers with the host.
 */

/** The frozen global the host puts its shared modules on (lib/theme-host/host-modules.ts). */
export interface ThemeHostGlobal {
  readonly sdkVersion: string
  /** Each shared specifier (shared-modules.json) and the host's own instance of it. */
  readonly modules: Readonly<Record<string, unknown>>
}

/** What register(host) receives: the SDK version and the one way to hand the host a slot. */
export interface ThemeRegistrationHost {
  readonly sdkVersion: string
  registerSlot(slot: string, component: unknown): void
}

/** A theme's default export, as defineTheme returns it. */
export interface ThemeDefinition {
  register(host: ThemeRegistrationHost): void
  unregister?(): void
}

declare global {
  interface Window {
    __AUTOMATOS_THEME_HOST__?: ThemeHostGlobal
  }
}
