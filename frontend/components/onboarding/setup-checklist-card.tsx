'use client'

import { CheckCircle2, Circle, ExternalLink, X, ListChecks } from 'lucide-react'
import { useWorkspace } from '@/components/workspace-provider'
import {
  useOnboardingChecklist,
  useUpdateChecklist,
  type ChecklistItem,
} from '@/hooks/use-onboarding-checklist'

/**
 * PRD-222 US-020 (W2·S4) — the post-setup "run & learn" checklist card.
 *
 * Rendered on chat AND the Command Center (the trial-balance-pill dual-surface
 * pattern): one server read-model, two mounts. Shows the outcome-framed next
 * steps whose completion is DERIVED server-side from real counts (connect a
 * second app · run your first mission · invite a teammate on multi-seat plans ·
 * take the matched Academy course). Dismissible; the dismissal is persisted in
 * the checklist doc server-side (a PATCH) — never a browser store (D8).
 *
 * Visible only once the workspace is past the build (stage powerup/completed)
 * and until dismissed. `className` lets each mount own its own positioning.
 */

const VISIBLE_STAGES = new Set(['powerup', 'completed'])

export function SetupChecklistCard({ className = '' }: { className?: string }) {
  const { workspace } = useWorkspace()
  const stage = workspace?.onboarding?.stage
  const active = !!stage && VISIBLE_STAGES.has(stage)

  const { data: checklist } = useOnboardingChecklist({ enabled: active })
  const update = useUpdateChecklist()

  if (!active || !checklist || checklist.dismissed) return null

  function dismiss() {
    update.mutate({ dismissed: true })
  }

  function openCourse(item: ChecklistItem) {
    // The Academy is a sibling repo with no completion signal back — checking
    // the item on click is the sanctioned manual exception (persisted server-side).
    if (!item.done) update.mutate({ academy_done: true })
  }

  return (
    <div
      data-testid="setup-checklist-card"
      className={[
        'bg-card/50 backdrop-blur border border-primary/20 rounded-xl p-4 space-y-3 max-w-md',
        className,
      ].filter(Boolean).join(' ')}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <ListChecks className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">
            Get the most out of your workspace
          </span>
        </div>
        <button
          type="button"
          onClick={dismiss}
          data-testid="setup-checklist-dismiss"
          className="text-muted-foreground hover:text-foreground transition-colors p-0.5"
          aria-label="Dismiss checklist"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <p data-testid="setup-checklist-progress" className="text-xs text-muted-foreground">
        {checklist.completed_count} of {checklist.total_count} done
      </p>

      <ul className="space-y-1.5">
        {checklist.items.map((item) => {
          const Icon = item.done ? CheckCircle2 : Circle
          const iconClass = item.done ? 'text-success' : 'text-muted-foreground'
          const content = (
            <span className="flex items-center gap-2">
              <Icon className={`w-4 h-4 shrink-0 ${iconClass}`} />
              <span
                className={
                  item.done
                    ? 'text-sm text-muted-foreground line-through'
                    : 'text-sm text-foreground/90'
                }
              >
                {item.label}
              </span>
            </span>
          )
          return (
            <li
              key={item.id}
              data-testid={`setup-checklist-item-${item.id}`}
              data-done={item.done ? 'true' : 'false'}
            >
              {item.href ? (
                <a
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => openCourse(item)}
                  data-testid={`setup-checklist-link-${item.id}`}
                  className="flex items-center justify-between gap-2 rounded-md px-1 py-0.5 hover:bg-secondary/40 transition-colors"
                >
                  {content}
                  <ExternalLink className="w-3 h-3 text-muted-foreground shrink-0" />
                </a>
              ) : (
                <div className="px-1 py-0.5">{content}</div>
              )}
            </li>
          )
        })}
      </ul>

      <p className="text-xs text-muted-foreground/80 leading-snug">
        Not sure? Ask Auto — he can do any of these for you.
      </p>
    </div>
  )
}
