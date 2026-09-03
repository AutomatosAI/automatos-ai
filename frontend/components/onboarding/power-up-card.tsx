'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import {
  Sparkles,
  KeyRound,
  X,
  ExternalLink,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'
import { isSaaS } from '@/lib/auth-edition'
import { keyPlaceholder, providersForByok, useProviderRegistry } from '@/hooks/use-provider-registry'

/**
 * PRD-222 US-013 (W1·S3) — the power-up card.
 *
 * Rendered in chat after the BOOM moment (onboarding.stage === 'powerup'), and
 * reused by US-014's exhausted banner (pass `embedded` to skip the stage gate).
 * Framed as continuation, never a paywall: "keep Auto running on your business".
 * The masked key posts to the existing BYOK endpoint (POST /api/keys), which
 * performs a live provider test — the card renders that result in-flow, the
 * provider's own error text and all (the badge never lies).
 *
 * Key handling: the raw key lives only in local input state and the request
 * body. It is never logged and never written to any client-side store
 * (localStorage / sessionStorage / global store); it is cleared from state on a
 * successful save.
 */

type Provider = string

interface KeySaveResponse {
  validation?: { valid: boolean; message: string } | null
}

export function PowerUpCard({ embedded = false }: { embedded?: boolean }) {
  const { workspace, refreshWorkspace } = useWorkspace()
  const trial = workspace?.onboarding?.trial ?? null

  const [dismissed, setDismissed] = useState(false)
  const [provider, setProvider] = useState<Provider>('openrouter')
  const [apiKey, setApiKey] = useState('')
  const [showOthers, setShowOthers] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null)

  // PRD-236: every BYO-key provider the registry knows, OpenRouter first
  // (static fallback = the pre-registry list when the call is unavailable).
  const registry = useProviderRegistry()
  const providerChips = useMemo(
    () => [
      { value: 'openrouter', label: 'OpenRouter' },
      ...providersForByok(registry)
        .filter((p) => p.slug !== 'openrouter')
        .map((p) => ({ value: p.slug, label: p.label })),
    ],
    [registry],
  )

  // In chat the card only appears at the powerup stage; the exhausted banner
  // (US-014) renders it embedded regardless of stage.
  // PRD-233 S7: the power-up / plan pitch is a hosted-edition surface.
  if (!isSaaS) return null
  if (!embedded && workspace?.onboarding?.stage !== 'powerup') return null
  if (dismissed) return null

  async function handleConnect() {
    const key = apiKey.trim()
    if (!key || submitting) return
    setSubmitting(true)
    setResult(null)
    try {
      // The raw key rides in the request body only — never persisted here.
      const res = await apiClient.post<KeySaveResponse>('/api/keys', {
        provider,
        api_key: key,
      })
      const validation = res?.validation
      if (validation?.valid) {
        setApiKey('') // drop the raw key from state the moment it is accepted
        setResult({ ok: true, message: validation.message || 'Key validated.' })
        await refreshWorkspace?.() // reflect the converted trial + unlocked catalog
      } else {
        setResult({
          ok: false,
          message: validation?.message || 'That key did not validate. Check it and try again.',
        })
      }
    } catch (err) {
      // Surface the failure without ever echoing the key.
      setResult({
        ok: false,
        message: err instanceof Error ? err.message : 'Could not reach the server. Try again.',
      })
    } finally {
      setSubmitting(false)
    }
  }

  const remaining = trial ? Math.max(0, trial.granted_usd - trial.spent_usd) : null

  return (
    <div
      data-testid="power-up-card"
      className="bg-card/50 backdrop-blur border border-primary/20 rounded-xl p-4 space-y-3 max-w-md"
    >
      {/* Header — continuation framing, not a paywall */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">
            Keep Auto running on your business
          </span>
        </div>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          data-testid="power-up-dismiss"
          className="text-muted-foreground hover:text-foreground transition-colors p-0.5"
          aria-label="Maybe later"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <p className="text-sm text-foreground/80 leading-snug">
        Connect your AI key (2 min) and Auto keeps working for you — no
        interruption.
      </p>

      {trial && trial.state !== 'converted' && (
        <p
          data-testid="power-up-trial"
          className={
            trial.state === 'warned' || trial.state === 'exhausted'
              ? 'text-xs font-medium text-amber-600 dark:text-amber-500'
              : 'text-xs text-muted-foreground'
          }
        >
          ${remaining?.toFixed(2)} of ${trial.granted_usd.toFixed(2)} trial credit left
        </p>
      )}

      {/* Recommendation on top — OpenRouter is the only top-level pick */}
      <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 space-y-2">
        <div className="flex items-center gap-2">
          <KeyRound className="w-3.5 h-3.5 text-primary" />
          <span className="text-xs font-medium text-foreground">
            Recommended: OpenRouter — one key, every model
          </span>
        </div>
        <a
          href="https://openrouter.ai/keys"
          target="_blank"
          rel="noopener noreferrer"
          data-testid="power-up-howto"
          className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
        >
          How to get an OpenRouter key
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>

      {/* Masked key entry */}
      <div className="space-y-2">
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          data-testid="power-up-key-input"
          autoComplete="off"
          spellCheck={false}
          placeholder={
            provider === 'openrouter' ? 'sk-or-…' : keyPlaceholder(registry, provider)
          }
          className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
        />
        <Button
          type="button"
          size="sm"
          onClick={handleConnect}
          disabled={submitting || !apiKey.trim()}
          data-testid="power-up-connect"
          className="w-full"
        >
          {submitting ? (
            <>
              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              Validating…
            </>
          ) : (
            `Connect ${provider === 'openrouter' ? 'OpenRouter' : 'key'}`
          )}
        </Button>
      </div>

      {/* Live validation result, rendered in-flow */}
      {result && (
        <div
          data-testid="power-up-result"
          className={
            result.ok
              ? 'rounded-md border border-emerald-500/30 bg-emerald-500/10 p-2.5 text-xs text-emerald-700 dark:text-emerald-400'
              : 'rounded-md border border-destructive/30 bg-destructive/10 p-2.5 text-xs text-destructive'
          }
        >
          <div className="flex items-start gap-1.5">
            {result.ok ? (
              <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            ) : (
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            )}
            <div>
              <p>{result.message}</p>
              {result.ok && (
                <p className="mt-1 text-foreground/70">
                  Your full model catalog is unlocked.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Other providers, collapsed beneath the recommendation */}
      <div>
        <button
          type="button"
          onClick={() => setShowOthers((v) => !v)}
          data-testid="power-up-others-toggle"
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <ChevronDown
            className={`w-3 h-3 transition-transform ${showOthers ? 'rotate-180' : ''}`}
          />
          I already use another provider (OpenAI, Anthropic, Google, NVIDIA…)
        </button>
        {showOthers && (
          <div className="mt-2 flex flex-wrap gap-1.5" data-testid="power-up-others">
            {providerChips.map((p) => (
              <button
                key={p.value}
                type="button"
                onClick={() => setProvider(p.value)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                  provider === p.value
                    ? 'border-primary/40 bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:text-foreground'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Skip-ahead + dismiss — the flow stays intact either way */}
      <div className="flex items-center justify-between pt-1">
        <Link
          href="/settings"
          data-testid="power-up-settings-link"
          className="text-xs text-muted-foreground hover:text-foreground hover:underline"
        >
          Prefer Settings → Credentials?
        </Link>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          Maybe later
        </button>
      </div>
    </div>
  )
}
