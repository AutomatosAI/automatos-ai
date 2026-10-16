// PRD-261 A-01: the globals the specs read in the page (inside page.evaluate and
// page.waitForFunction), typed once.

export interface ThemeHostGlobalSeen {
  readonly sdkVersion: string
  readonly modules: Readonly<Record<string, Readonly<Record<string, unknown>>>>
}

export interface ShimComparison {
  readonly spec: string
  readonly missing: string[]
  readonly differing: string[]
}

export interface ShimCheck {
  readonly results: ShimComparison[]
  readonly error: string | null
}

export type ProbeOutcome = 'loaded' | 'refused'

declare global {
  interface Window {
    __AUTOMATOS_THEME_HOST__?: ThemeHostGlobalSeen
    __shimSpecs?: string[]
    __shimCheck?: ShimCheck
    __probeResult?: Record<string, ProbeOutcome>
  }
}
