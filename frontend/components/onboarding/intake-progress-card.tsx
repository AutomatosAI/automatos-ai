'use client'

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Loader2,
  Network,
  Sparkles,
  Upload,
} from 'lucide-react'
import {
  useWizardProgress,
  type WizardStage,
} from '@/hooks/use-wizard-progress'

/**
 * PRD-222 US-015 (W1·S8) — the intake progress card.
 *
 * A compact chat status card for the intake pipeline Auto kicks off with
 * `platform_scan_business_site`: scan → scrape → ingest → graph → profile. It
 * REUSES the wizard's SSE hook (`useWizardProgress`) verbatim — the same
 * fetch+ReadableStream feed, auth headers, dedupe and auto-reconnect the wizard
 * consumes — so there is one streaming implementation, not two. The wizard's own
 * consumption is untouched; both surfaces read the same hook.
 *
 * Terminal states are honest:
 *   - complete → a "here's what I learned" handoff line (Auto takes it from here)
 *   - failed / stream error → the failure, plus an "upload docs instead" fallback.
 */

// The five user-facing pipeline stages, in order. `graphify` is the hook's stage
// id; we label it "Graph" to match the PRD's scan → scrape → ingest → graph →
// profile wording.
const PIPELINE: { stage: WizardStage; label: string; icon: typeof FileText }[] = [
  { stage: 'scan', label: 'Scan', icon: Network },
  { stage: 'scrape', label: 'Scrape', icon: FileText },
  { stage: 'ingest', label: 'Ingest', icon: Activity },
  { stage: 'graphify', label: 'Graph', icon: Sparkles },
  { stage: 'profile', label: 'Profile', icon: CheckCircle2 },
]

const ORDER: WizardStage[] = PIPELINE.map((p) => p.stage)

interface IntakeProgressCardProps {
  /** The intake profile started by `platform_scan_business_site`. */
  profileId: string | null
  /** Open the stream. Defaults to true; the hook stops itself on a terminal event. */
  active?: boolean
  /** Skip-ahead when the scan can't finish — wire to the doc-upload affordance. */
  onUploadDocs?: () => void
}

export function IntakeProgressCard({
  profileId,
  active = true,
  onUploadDocs,
}: IntakeProgressCardProps) {
  const { events, state, latest } = useWizardProgress({ profileId, active })

  if (!profileId) return null

  const reached = new Set<WizardStage>(events.map((e) => e.stage))
  const currentStage = latest?.stage ?? null
  const failed = state === 'failed' || state === 'error'
  const complete = state === 'complete'

  return (
    <div
      data-testid="intake-progress-card"
      data-state={state}
      className="bg-card/50 backdrop-blur border border-primary/20 rounded-xl p-4 space-y-3 max-w-md"
    >
      {/* Header */}
      <div className="flex items-center gap-2">
        {complete ? (
          <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
        ) : failed ? (
          <AlertTriangle className="w-4 h-4 text-destructive shrink-0" />
        ) : (
          <Loader2 className="w-4 h-4 text-primary shrink-0 animate-spin" />
        )}
        <span className="text-sm font-medium text-foreground">
          {complete
            ? 'Finished reading your business'
            : failed
              ? "Couldn't finish reading your site"
              : 'Reading your business'}
        </span>
      </div>

      {/* Pipeline stage row */}
      <div className="flex flex-wrap items-center gap-1.5" data-testid="intake-stages">
        {PIPELINE.map(({ stage, label, icon: Icon }) => {
          const isReached = reached.has(stage)
          const isActive = currentStage === stage && state === 'streaming'
          const isDone =
            complete ||
            (isReached &&
              !isActive &&
              currentStage !== null &&
              ORDER.indexOf(currentStage) > ORDER.indexOf(stage))
          return (
            <div
              key={stage}
              data-testid={`intake-stage-${stage}`}
              data-status={isActive ? 'active' : isDone ? 'done' : isReached ? 'reached' : 'pending'}
              className={[
                'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium',
                isActive
                  ? 'border-primary bg-primary/10 text-primary'
                  : isDone
                    ? 'border-success/40 bg-success/10 text-success'
                    : 'border-border/40 bg-secondary/20 text-muted-foreground',
              ].join(' ')}
            >
              {isActive ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Icon className="w-3 h-3" />
              )}
              {label}
            </div>
          )
        })}
      </div>

      {/* Live status line while streaming */}
      {!complete && !failed && latest && (
        <p data-testid="intake-status" className="text-xs text-muted-foreground leading-snug">
          {latest.message}
        </p>
      )}

      {/* Terminal: profile ready — hand off to Auto */}
      {complete && (
        <p data-testid="intake-handoff" className="text-sm text-foreground/80 leading-snug">
          Here&apos;s what I learned about your business — I&apos;ll use it from here.
        </p>
      )}

      {/* Terminal: failed — honest error + upload-docs fallback */}
      {failed && (
        <div className="space-y-2">
          <p data-testid="intake-error" className="text-sm text-destructive/90 leading-snug">
            {latest?.message ||
              (state === 'failed'
                ? 'The intake pipeline stopped before it finished.'
                : 'The progress stream dropped before it finished.')}
          </p>
          <button
            type="button"
            onClick={onUploadDocs}
            data-testid="intake-upload-fallback"
            className="inline-flex items-center gap-1.5 text-xs text-primary hover:underline"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload your docs instead and I&apos;ll work from those
          </button>
        </div>
      )}
    </div>
  )
}
