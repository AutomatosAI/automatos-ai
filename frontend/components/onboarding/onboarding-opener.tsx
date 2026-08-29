'use client'

import { useWorkspace } from '@/components/workspace-provider'

/**
 * PRD-222 US-012 — Auto's opening move.
 *
 * On first chat load for a brand-new workspace (`onboarding.stage ===
 * 'not_started'`) Auto makes the first move: a short canned greeting that asks
 * the first onboarding question and invites the user to just start typing. Any
 * reply flows through the server-side `OnboardingSection` (US-004), which drives
 * the stage machine (`questions → teach → proposal → …`) from there — this
 * component only renders the opener, it never writes onboarding state.
 *
 * Self-guarding: renders nothing unless the workspace is at `not_started`, so it
 * is safe to mount unconditionally in the empty chat state. It reads the stage
 * from the server snapshot on `workspace-provider` (never localStorage — D8).
 */
export function OnboardingOpener() {
  const { workspace } = useWorkspace()

  if (workspace?.onboarding?.stage !== 'not_started') {
    return null
  }

  return (
    <div
      data-testid="onboarding-opener"
      className="w-full max-w-3xl md:max-w-4xl text-center mb-8"
    >
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground/90">
        Hi — I&apos;m <span className="gradient-text">Auto</span>.
      </h1>
      <p className="mt-4 text-lg text-muted-foreground">
        Let&apos;s set your workspace up together. To start —{' '}
        <span className="text-foreground/90">what&apos;s your business?</span>
      </p>
      <p className="mt-2 text-sm text-muted-foreground">
        Tell me below, or just start typing — whatever&apos;s easier.
      </p>
    </div>
  )
}
