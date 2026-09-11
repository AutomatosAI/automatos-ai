'use client'

/**
 * The operator console's plan dropdown (admin/workspaces).
 *
 * Changing a tier changes what a LIVE tenant can see and do, so the control is
 * deliberately asymmetric:
 *
 *   * an UPGRADE (nothing is taken away) applies on selection — that is the
 *     common case, and the reason this control exists;
 *   * a change that REMOVES something — a nav surface the tier no longer
 *     exposes, or a seat cap below the members already in the workspace —
 *     stops for confirmation and names exactly what it removes first.
 *
 * The backend never writes a spend ceiling from this path
 * (``assign_plan(..., with_budget=False)``), so no tier change here can hand a
 * tenant a ``check_budget`` throttle it did not already have.
 */

import { useMemo, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { apiClient } from '@/lib/api-client'

export interface PlanTier {
  name: string
  display_name: string | null
  assignable: boolean
  coming_soon: boolean
  seats: number | null
  max_agents: number | null
  marketplace_depth: number
  families: Record<string, boolean>
  nav: Record<string, boolean>
}

interface PlanChangeResponse {
  plan: string
  previous_plan: string | null
  warnings: string[]
}

interface WorkspacePlanSelectProps {
  workspaceId: string
  workspaceName: string
  plan: string | null
  membersCount: number
  tiers: PlanTier[]
  disabled?: boolean
  onChanged: () => void | Promise<void>
  onError: (message: string) => void
}

/** Nav keys the exposure profile actually gates, in the rail's own words. */
const NAV_LABELS: Record<string, string> = {
  team: 'Team Management',
  analytics: 'Analytics',
}

/** Capability families, as Auto's tool surface trims them. */
const FAMILY_LABELS: Record<string, string> = {
  codegraph: 'CodeGraph tools',
  nl2sql: 'Data queries & analytics tools',
  team: 'Team tools',
  voice: 'Voice',
}

function tierLabel(tier: PlanTier): string {
  const name = tier.display_name || tier.name
  return tier.coming_soon ? `${name} (coming soon)` : name
}

/**
 * What moving `from` → `to` adds and removes.
 *
 * Visibility mirrors `lib/nav-exposure.isNavItemVisible`: a key is visible
 * unless it is explicitly `false`. A tier that declares no policy at all
 * (enterprise) is therefore unrestricted, matching the backend.
 */
function surfaceDelta(from: PlanTier | undefined, to: PlanTier | undefined) {
  const gained: string[] = []
  const lost: string[] = []
  if (!from || !to) return { gained, lost }

  const compare = (
    beforeMap: Record<string, boolean>,
    afterMap: Record<string, boolean>,
    labels: Record<string, string>,
  ) => {
    const keys = new Set([...Object.keys(beforeMap), ...Object.keys(afterMap)])
    keys.forEach((key) => {
      const before = beforeMap[key] !== false
      const after = afterMap[key] !== false
      const label = labels[key] || key
      if (!before && after) gained.push(label)
      if (before && !after) lost.push(label)
    })
  }

  compare(from.nav || {}, to.nav || {}, NAV_LABELS)
  compare(from.families || {}, to.families || {}, FAMILY_LABELS)
  return { gained, lost }
}

/**
 * Would this tier's seat cap sit below the members already in the workspace?
 *
 * Mirrors the enforcement in ``core/workspaces/invitations.py`` exactly: -1 is
 * the only unlimited sentinel there, so seats=0 is a hard block, not
 * "unlimited". (plan_tiers documents 0-means-unlimited for max_agents and
 * watcher_limit — a different key with a different rule. Do not borrow it.)
 */
function isSeatSqueeze(tier: PlanTier | undefined, membersCount: number): boolean {
  if (typeof tier?.seats !== 'number') return false
  return tier.seats !== -1 && tier.seats < membersCount
}

export function WorkspacePlanSelect({
  workspaceId,
  workspaceName,
  plan,
  membersCount,
  tiers,
  disabled = false,
  onChanged,
  onError,
}: WorkspacePlanSelectProps) {
  const [pending, setPending] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const current = useMemo(
    () => tiers.find((t) => t.name === plan),
    [tiers, plan],
  )
  const target = useMemo(
    () => tiers.find((t) => t.name === pending),
    [tiers, pending],
  )

  const delta = useMemo(() => surfaceDelta(current, target), [current, target])

  // Mirrors core/workspaces/invitations.py, where -1 is the ONLY "unlimited"
  // sentinel — a tier tuned to seats=0 blocks every invite and must still warn.
  const seatSqueeze = isSeatSqueeze(target, membersCount)

  async function apply(nextPlan: string) {
    setSaving(true)
    try {
      // apiClient.patch types its body as `any`; passing an object literal
      // straight into request()'s RequestInit would be a type error.
      // The response is typed for the contract, not consumed: the confirm modal
      // has already shown the operator any warning this call can return.
      await apiClient.patch<PlanChangeResponse>(
        `/api/admin/workspaces/${workspaceId}/plan`,
        { plan: nextPlan },
      )
      setPending(null)
      await onChanged()
    } catch (err: any) {
      onError(err?.message || `Failed to move ${workspaceName} to ${nextPlan}`)
      setPending(null)
    } finally {
      setSaving(false)
    }
  }

  function handleSelect(nextPlan: string) {
    if (!nextPlan || nextPlan === plan) return
    const next = tiers.find((t) => t.name === nextPlan)
    const removes = surfaceDelta(current, next)
    const squeezes = isSeatSqueeze(next, membersCount)
    // Pure upgrade — nothing is taken away, so don't make the operator confirm.
    if (removes.lost.length === 0 && !squeezes) {
      void apply(nextPlan)
      return
    }
    setPending(nextPlan)
  }

  // A plan string with no matching tier (a stray or renamed value) still has to
  // render as the current selection rather than showing an empty control.
  const unknownPlan = plan && !current

  return (
    <>
      <Select
        value={plan || undefined}
        onValueChange={handleSelect}
        disabled={disabled || saving || tiers.length === 0}
      >
        <SelectTrigger className="h-7 w-[124px] text-xs capitalize">
          {saving ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <SelectValue placeholder="—" />
          )}
        </SelectTrigger>
        <SelectContent>
          {unknownPlan && (
            <SelectItem value={plan} disabled className="text-xs">
              {plan} (unknown tier)
            </SelectItem>
          )}
          {tiers.map((tier) => (
            <SelectItem
              key={tier.name}
              value={tier.name}
              // Non-assignable tiers stay visible so a workspace already on one
              // reads correctly, but they cannot be selected.
              disabled={!tier.assignable && tier.name !== plan}
              className="text-xs"
            >
              {tierLabel(tier)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {pending && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => !saving && setPending(null)}
        >
          <Card className="max-w-md w-full" onClick={(e) => e.stopPropagation()}>
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center gap-2 text-warning">
                <AlertTriangle className="h-5 w-5" />
                <h2 className="text-lg font-semibold">
                  Move to {target ? tierLabel(target) : pending}?
                </h2>
              </div>

              <p className="text-sm text-muted-foreground">
                <span className="font-semibold text-foreground">{workspaceName}</span>{' '}
                moves from {current ? tierLabel(current) : plan || 'no plan'} to{' '}
                {target ? tierLabel(target) : pending}. This takes effect immediately.
              </p>

              {delta.lost.length > 0 && (
                <div className="text-sm">
                  <p className="font-medium text-destructive mb-1">Removes</p>
                  <ul className="list-disc pl-5 text-muted-foreground space-y-0.5">
                    {delta.lost.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {delta.gained.length > 0 && (
                <div className="text-sm">
                  <p className="font-medium text-success mb-1">Adds</p>
                  <ul className="list-disc pl-5 text-muted-foreground space-y-0.5">
                    {delta.gained.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {seatSqueeze && (
                <p className="text-sm text-warning">
                  This workspace has {membersCount} active members but{' '}
                  {target?.display_name || pending} allows {target?.seats}. Nobody is
                  removed — the cap applies to the next invitation.
                </p>
              )}

              <p className="text-xs text-muted-foreground">
                No spend ceiling is written by this action.
              </p>

              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => setPending(null)}
                  disabled={saving}
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => apply(pending)}
                  disabled={saving}
                  className="bg-warning hover:bg-warning/80 text-white"
                >
                  {saving ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    'Change plan'
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  )
}
