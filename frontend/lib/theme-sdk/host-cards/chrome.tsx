/**
 * PRD-261 A-12a — the SDK's chrome cards: `cards.HeaderUtilities`, `cards.TrialExhaustedBanner` and
 * `cards.FirstRunNudge`.
 *
 * HeaderUtilities is the Studio header's utilities cluster (components/layout/studio-header.tsx):
 * the credit balance (hosted edition, PRD-265), Help and the docs, notifications, the theme menu
 * (Style, with the way back to Classic and Studio, and Tone) and the profile control by edition.
 * Every frame must render it (PRD §4.4), so the theme menu and the account are always on screen;
 * lib/theme-sdk/__tests__/header-utilities.test.tsx checks an installed theme's frame does. It has
 * no safe-mode entry: safe mode is the /safe-mode page or `?safe_mode=themes`. The banner
 * and the nudge are the host's own: the paused state once the workspace's credits are used up
 * (PRD-265; the card keeps its SDK name) with its power-up card, and the local edition's "add an
 * LLM key" nudge.
 */
import type { ReactElement } from 'react'

import { StudioHeaderUtilities } from '@/components/layout/studio-header'
import { FirstRunNudge as HostFirstRunNudge } from '@/components/local/first-run-nudge'
import { TrialExhaustedBanner as HostTrialExhaustedBanner } from '@/components/onboarding/trial-exhausted-banner'

export interface TrialExhaustedBannerProps {
  /** The typed error a refused request returned (`credits_exhausted`), before the workspace's balance catches up. */
  readonly errorCode?: string | null
}

/** The host's header utilities, which every frame renders. */
export function HeaderUtilities(): ReactElement {
  return <StudioHeaderUtilities />
}

/** The host's banner for used-up credits (work on Automatos's AI keys paused); it draws nothing while work runs. */
export function TrialExhaustedBanner({ errorCode }: TrialExhaustedBannerProps): ReactElement {
  return <HostTrialExhaustedBanner errorCode={errorCode} />
}

/** The local edition's first-run nudge; it draws nothing in SaaS, once dismissed, or once a key exists. */
export function FirstRunNudge(): ReactElement {
  return <HostFirstRunNudge />
}
