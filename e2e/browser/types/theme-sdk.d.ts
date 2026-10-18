// PRD-261 A-01, A-04a: the types a test theme sees for '@automatos/theme-sdk': its version,
// defineTheme and the registration host (frontend/lib/theme-host/types.ts ThemeRegistrationHost).
// The host's runtime is frontend/lib/theme-host/sdk.ts; the SDK proper, with its published types,
// is PRD-261 A-10.
declare module '@automatos/theme-sdk' {
  import type { ComponentType, ReactNode } from 'react'

  export const SDK_VERSION: string

  /** Slot 'frame@1': the page, MainLayout's full-bleed flag and Auto's widget. */
  export interface FrameProps {
    readonly children?: ReactNode
    readonly fullBleed?: boolean
    readonly autoWidget?: ReactNode
  }

  export interface ThemeHost {
    readonly sdkVersion: string
    readonly hostVersion: string
    /** The frame drawn around the page; the host serves the other slots by name. */
    registerSlot(slot: 'frame@1', component: ComponentType<FrameProps>): void
    registerSlot(slot: string, component: ComponentType<never>): void
    /** One of the core pages a theme may draw its own version of (the Q3 list). */
    registerPage(route: string, component: ComponentType<never>): void
    /** A page of the theme's own, served at /x/<path>. */
    registerRoute(path: string, component: ComponentType<never>): void
  }

  export interface ThemeDefinition {
    register(host: ThemeHost): void
    unregister?(): void
  }

  export function defineTheme(definition: ThemeDefinition): Readonly<ThemeDefinition>
}
