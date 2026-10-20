/**
 * PRD-261 A-12a — the SDK's chrome cards: `cards.HeaderUtilities`, `cards.TrialExhaustedBanner` and
 * `cards.FirstRunNudge`.
 *
 * HeaderUtilities is the Studio header's utilities cluster (components/layout/studio-header.tsx):
 * Help and the docs, notifications, the theme menu and the profile control by edition. Every frame
 * must render it (PRD §4.4), so the way back to the theme menu, safe mode's entry and the account
 * are always on screen; lib/theme-sdk/__tests__/header-utilities.test.tsx checks builtin Play's
 * frame does. The banner and the nudge are the host's own: the trial's exhausted state with its
 * power-up card, and the local edition's "add an LLM key" nudge.
 */
import type { ReactElement } from 'react'

import { StudioHeaderUtilities } from '@/components/layout/studio-header'
import { FirstRunNudge as HostFirstRunNudge } from '@/components/local/first-run-nudge'
import { TrialExhaustedBanner as HostTrialExhaustedBanner } from '@/components/onboarding/trial-exhausted-banner'

export interface TrialExhaustedBannerProps {
  /** The typed error a blocked request returned (`trial_exhausted`), before the workspace's snapshot catches up. */
  readonly errorCode?: string | null
}

/** The host's header utilities, which every frame renders. */
export function HeaderUtilities(): ReactElement {
  return <StudioHeaderUtilities />
}

/** The host's banner for an exhausted trial; it draws nothing while the trial is not exhausted. */
export function TrialExhaustedBanner({ errorCode }: TrialExhaustedBannerProps): ReactElement {
  return <HostTrialExhaustedBanner errorCode={errorCode} />
}

/** The local edition's first-run nudge; it draws nothing in SaaS, once dismissed, or once a key exists. */
export function FirstRunNudge(): ReactElement {
  return <HostFirstRunNudge />
}
