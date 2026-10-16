// PRD-261 A-01: the types a theme sees for '@automatos/theme-sdk' in the spike: its version and
// defineTheme. The host's runtime is frontend/lib/theme-host/sdk.ts; the SDK proper, with its
// published types, is PRD-261 A-10.
declare module '@automatos/theme-sdk' {
  import type { ComponentType, ReactNode } from 'react'

  export const SDK_VERSION: string

  export interface ThemeHost {
    readonly sdkVersion: string
    /** The spike's one slot: the frame drawn around the page. */
    registerSlot(slot: 'frame@1', component: ComponentType<{ children?: ReactNode }>): void
  }

  export interface ThemeDefinition {
    register(host: ThemeHost): void
    unregister?(): void
  }

  export function defineTheme(definition: ThemeDefinition): Readonly<ThemeDefinition>
}
