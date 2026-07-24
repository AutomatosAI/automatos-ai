'use client'

/**
 * PolicyPane — PRD-196 S4 (P2-15, governance I.3). Steer the plane without
 * hand-writing JSON: posture (Balanced/Strict/Permissive), per-risk auto/ask
 * overrides, agents-inherit-admin, and the spend/token budget.
 *
 * Honest UI over silent placebo: the pane links the S3 policy-plane indicator
 * and says plainly that these settings take effect only when enforcement is on
 * (the plane ships default-OFF until PRD-192 flips it).
 */

import { useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import {
  usePolicy,
  useUpdatePolicy,
  useBudget,
  useUpdateBudget,
  useGovernanceStatus,
} from '@/hooks/use-governance'

const POSTURES = [
  { value: 'balanced', label: 'Balanced', hint: 'Auto low-risk; ask on publish / external / destructive.' },
  { value: 'strict', label: 'Strict', hint: 'Ask on every write, external, and destructive action.' },
  { value: 'permissive', label: 'Permissive', hint: 'Ask only on destructive + over-budget.' },
]

const RISK_CLASSES = [
  { key: 'read', label: 'Read' },
  { key: 'internal_write', label: 'Internal write' },
  { key: 'publish', label: 'Publish' },
  { key: 'external_side_effect', label: 'External side-effect' },
  { key: 'destructive', label: 'Destructive' },
]

const WINDOWS = ['day', 'month', 'all']

type OverrideValue = '' | 'auto' | 'ask'

export function PolicyPane() {
  const { data: policy } = usePolicy()
  const { data: budget } = useBudget()
  const { data: status } = useGovernanceStatus()
  const updatePolicy = useUpdatePolicy()
  const updateBudget = useUpdateBudget()

  const enforcing = status?.policy_plane?.enforcing ?? null

  const [posture, setPosture] = useState('balanced')
  const [inherit, setInherit] = useState(false)
  const [overrides, setOverrides] = useState<Record<string, OverrideValue>>({})
  const [maxCost, setMaxCost] = useState<string>('')
  const [maxTokens, setMaxTokens] = useState<string>('')
  const [budgetWindow, setBudgetWindow] = useState<string>('day')

  useEffect(() => {
    if (policy) {
      setPosture(policy.posture)
      setInherit(policy.agents_inherit_admin)
      setOverrides((policy.route_overrides as Record<string, OverrideValue>) ?? {})
    }
  }, [policy])

  useEffect(() => {
    if (budget) {
      setMaxCost(budget.max_cost_usd != null ? String(budget.max_cost_usd) : '')
      setMaxTokens(budget.max_total_tokens != null ? String(budget.max_total_tokens) : '')
      setBudgetWindow(budget.window || 'day')
    }
  }, [budget])

  const savePolicy = async () => {
    const cleanOverrides: Record<string, 'auto' | 'ask'> = {}
    for (const [k, v] of Object.entries(overrides)) {
      if (v === 'auto' || v === 'ask') cleanOverrides[k] = v
    }
    try {
      await updatePolicy.mutateAsync({
        posture,
        agents_inherit_admin: inherit,
        route_overrides: cleanOverrides,
      })
      toast.success('Policy saved')
    } catch {
      toast.error('Failed to save policy')
    }
  }

  const saveBudget = async () => {
    const body: { max_cost_usd?: number; max_total_tokens?: number; window?: string } = {
      window: budgetWindow,
    }
    if (maxCost.trim() !== '') body.max_cost_usd = Number(maxCost)
    if (maxTokens.trim() !== '') body.max_total_tokens = Number(maxTokens)
    try {
      await updateBudget.mutateAsync(body)
      toast.success('Budget saved')
    } catch {
      toast.error('Failed to save budget')
    }
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Honest banner: settings are inert until the plane is enforcing. */}
      {enforcing === false && (
        <div
          className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs dark:border-amber-800 dark:bg-amber-950/40"
          role="note"
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
          <span className="text-amber-800 dark:text-amber-300">
            The policy plane is <strong>OFF</strong> — these settings are saved but take effect only
            when enforcement is enabled. See the “Policy plane” tile above.
          </span>
        </div>
      )}

      {/* Posture */}
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Posture</h3>
        <div className="flex flex-col gap-1.5">
          {POSTURES.map((p) => (
            <label key={p.value} className="flex items-start gap-2 text-sm">
              <input
                type="radio"
                name="posture"
                value={p.value}
                checked={posture === p.value}
                onChange={() => setPosture(p.value)}
                className="mt-0.5"
              />
              <span>
                <span className="font-medium">{p.label}</span>
                <span className="block text-xs text-muted-foreground">{p.hint}</span>
              </span>
            </label>
          ))}
        </div>
      </section>

      {/* Per-risk overrides */}
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Per-risk overrides
        </h3>
        <div className="flex flex-col gap-1.5">
          {RISK_CLASSES.map((r) => (
            <div key={r.key} className="flex items-center justify-between gap-2 text-sm">
              <span>{r.label}</span>
              <select
                aria-label={`Override for ${r.label}`}
                value={overrides[r.key] ?? ''}
                onChange={(e) =>
                  setOverrides({ ...overrides, [r.key]: e.target.value as OverrideValue })
                }
                className="rounded border border-border bg-background px-2 py-0.5 text-xs"
              >
                <option value="">Posture default</option>
                <option value="auto">Auto</option>
                <option value="ask">Ask</option>
              </select>
            </div>
          ))}
        </div>
        <label className="flex items-center gap-2 text-sm mt-1">
          <input type="checkbox" checked={inherit} onChange={(e) => setInherit(e.target.checked)} />
          <span>
            Agents inherit admin from the workspace owner
            <span className="block text-xs text-muted-foreground">
              F014 — off by default; when off, admin-only actions require the caller’s own admin role.
            </span>
          </span>
        </label>
        <div>
          <Button size="sm" disabled={updatePolicy.isLoading} onClick={savePolicy}>
            {updatePolicy.isLoading ? 'Saving…' : 'Save policy'}
          </Button>
        </div>
      </section>

      {/* Budget */}
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Budget</h3>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-muted-foreground">Spend ceiling (USD)</span>
            <input
              type="number"
              min="0"
              step="0.01"
              value={maxCost}
              onChange={(e) => setMaxCost(e.target.value)}
              placeholder="no ceiling"
              className="rounded border border-border bg-background px-2 py-1"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-muted-foreground">Token ceiling</span>
            <input
              type="number"
              min="0"
              step="1"
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
              placeholder="no ceiling"
              className="rounded border border-border bg-background px-2 py-1"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-muted-foreground">Window</span>
            <select
              value={budgetWindow}
              onChange={(e) => setBudgetWindow(e.target.value)}
              className="rounded border border-border bg-background px-2 py-1"
            >
              {WINDOWS.map((w) => (
                <option key={w} value={w}>
                  {w}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div>
          <Button size="sm" disabled={updateBudget.isLoading} onClick={saveBudget}>
            {updateBudget.isLoading ? 'Saving…' : 'Save budget'}
          </Button>
        </div>
      </section>
    </div>
  )
}
