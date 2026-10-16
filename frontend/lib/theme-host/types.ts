/**
 * PRD-261 A-01 — the theme host's types: the global a theme's shims read, what a theme
 * registers with the host, the spike's manifest and preset, and what a load returns.
 */
import type { ComponentType, ReactNode } from 'react'

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

/** Why a theme did not start, or stopped: each reason has its own notice text. */
export type ThemeFailureReason = 'manifest' | 'engines' | 'load' | 'register' | 'look' | 'render'

/** What the host is doing: no theme here, loading one, showing one, fell back, or safe mode. */
export type ThemeHostStatus = 'off' | 'loading' | 'active' | 'failed' | 'safe-mode'

/** The component a theme registers for the frame slot; it draws its frame around `children`. */
export type FrameComponent = ComponentType<{ children?: ReactNode }>

/** A checked theme.json. The three file fields are absolute URLs under the theme's folder. */
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

export interface ThemeLoadOk {
  readonly ok: true
  readonly manifest: SpikeManifest
  readonly frame: FrameComponent
  /** Removes the theme's class, stylesheet and variables, then runs its unregister; safe to call twice. */
  readonly cleanup: () => void
}

export interface ThemeLoadFailure {
  readonly ok: false
  readonly reason: ThemeFailureReason
  readonly detail: string
}

export type ThemeLoadResult = ThemeLoadOk | ThemeLoadFailure

declare global {
  interface Window {
    __AUTOMATOS_THEME_HOST__?: ThemeHostGlobal
  }
}
