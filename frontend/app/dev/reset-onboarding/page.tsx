'use client'

import { useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { RotateCcw, AlertTriangle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'
import { resetAllTours, resetOnboarding } from '@/lib/shepherd/tour-storage'

/**
 * PRD-222 US-016 (W1·S10 / D9) — DEV/OPS reset console. UNLINKED from all nav:
 * reachable only by typing /dev/reset-onboarding. Lets the operator rewind
 * onboarding in ONE workspace with a single alias account instead of
 * provisioning/deleting a workspace per attempt.
 *
 * Posts the three flags to POST /api/workspaces/current/onboarding/reset via a
 * raw fetch (so the 404-when-disabled response is read from the status code, not
 * an opaque thrown error), then clears the tour/onboarding localStorage through
 * the existing shepherd tour-storage helpers and drops the operator back into
 * the empty chat so the flow re-fires from not_started.
 */
export default function ResetOnboardingPage() {
  const { user } = useUser()
  const router = useRouter()
  const { workspace, refreshWorkspace } = useWorkspace()

  const [resetTrial, setResetTrial] = useState(false)
  const [wipeBuilt, setWipeBuilt] = useState(false)
  const [wipeCredentials, setWipeCredentials] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [disabled, setDisabled] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const stage = workspace?.onboarding?.stage ?? 'unknown'
  const trial = workspace?.onboarding?.trial ?? null
  // A converted workspace with its key wiped falls through to unmetered platform
  // resolution — surface that trap before the operator pulls the trigger.
  const credentialsWarning = wipeCredentials && !resetTrial

  async function handleReset() {
    setSubmitting(true)
    setError(null)
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(await apiClient.getAuthHeaders()),
      }
      const res = await fetch(
        `${apiClient.getBaseUrl()}/api/workspaces/current/onboarding/reset`,
        {
          method: 'POST',
          headers,
          body: JSON.stringify({
            reset_trial: resetTrial,
            wipe_built: wipeBuilt,
            wipe_credentials: wipeCredentials,
          }),
        },
      )

      if (res.status === 404) {
        // Feature flag off — the endpoint is unadvertised. Show the plain state.
        setDisabled(true)
        return
      }
      if (!res.ok) {
        setError(`Reset failed (HTTP ${res.status}).`)
        return
      }

      // Clear the client-side tour/onboarding flags via the shepherd helpers so
      // Auto's tours re-evaluate cleanly against the fresh server state.
      const userId = user?.id
      if (userId) {
        resetAllTours(userId)
        resetOnboarding(userId)
      }
      await refreshWorkspace?.()
      router.push('/chat')
    } catch (e: any) {
      setError(e?.message ?? 'Reset failed.')
    } finally {
      setSubmitting(false)
    }
  }

  if (disabled) {
    return (
      <div className="mx-auto max-w-lg p-8" data-testid="reset-disabled">
        <h1 className="text-lg font-semibold">Onboarding reset is disabled</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Set <code>ONBOARDING_RESET_ENABLED=true</code> on this environment to arm
          the dev reset endpoint. It stays off (and unadvertised) everywhere else.
        </p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-lg p-8">
      <div className="flex items-center gap-2">
        <RotateCcw className="h-5 w-5" />
        <h1 className="text-lg font-semibold">Reset onboarding (dev)</h1>
      </div>
      <p className="mt-1 text-sm text-muted-foreground">
        Rewinds THIS workspace so Auto-led onboarding runs again from the top.
        Dev/ops only — unlinked from navigation.
      </p>

      <div className="mt-6 rounded-lg border p-4 text-sm">
        <div>
          Current stage:{' '}
          <span className="font-mono font-medium" data-testid="reset-stage">
            {stage}
          </span>
        </div>
        <div className="mt-1" data-testid="reset-trial">
          Trial:{' '}
          {trial ? (
            <span className="font-mono">
              {trial.state} · ${Number(trial.spent_usd ?? 0).toFixed(2)} of $
              {Number(trial.granted_usd ?? 0).toFixed(2)}
            </span>
          ) : (
            <span className="text-muted-foreground">none</span>
          )}
        </div>
      </div>

      <div className="mt-6 space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="reset-trial-toggle">Reset trial (re-grant a fresh $0 trial)</Label>
          <Switch
            id="reset-trial-toggle"
            data-testid="toggle-reset-trial"
            checked={resetTrial}
            onCheckedChange={setResetTrial}
          />
        </div>
        <div className="flex items-center justify-between">
          <Label htmlFor="wipe-built-toggle">Wipe what onboarding built (agents, missions, docs…)</Label>
          <Switch
            id="wipe-built-toggle"
            data-testid="toggle-wipe-built"
            checked={wipeBuilt}
            onCheckedChange={setWipeBuilt}
          />
        </div>
        <div className="flex items-center justify-between">
          <Label htmlFor="wipe-credentials-toggle">Wipe this workspace's credentials</Label>
          <Switch
            id="wipe-credentials-toggle"
            data-testid="toggle-wipe-credentials"
            checked={wipeCredentials}
            onCheckedChange={setWipeCredentials}
          />
        </div>
      </div>

      {credentialsWarning && (
        <div
          className="mt-4 flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-300"
          data-testid="reset-credentials-warning"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            Wiping credentials without resetting the trial leaves a converted
            workspace with no key — requests fall through to unmetered platform
            resolution. Turn on “Reset trial” too, or leave credentials in place.
          </span>
        </div>
      )}

      {error && (
        <div className="mt-4 text-sm text-red-600 dark:text-red-400" data-testid="reset-error">
          {error}
        </div>
      )}

      <Button
        className="mt-6 w-full"
        variant="destructive"
        disabled={submitting}
        onClick={handleReset}
        data-testid="reset-submit"
      >
        {submitting ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Resetting…
          </>
        ) : (
          'Reset onboarding'
        )}
      </Button>
    </div>
  )
}
