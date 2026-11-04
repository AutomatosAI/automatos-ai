'use client'

/**
 * PRD-268 §6.9 — "Pause suggestions", the signed-in person's own, for every role: GET and PUT /api/proactive/me
 * `{paused}`. Shown at once and put back, with the server's sentence, when the server refuses. A paused person gets
 * no suggestions and no suggestion notifications; the brief still comes unless the digest is off.
 */
import { useChangeMyProactiveSettings, useMyProactiveSettings } from '@/hooks/use-proactive-settings'
import { LOADING, PAUSE_LABEL, PAUSE_LINE } from './copy'
import { FailureLine, SwitchRow } from './parts'

export function PauseSuggestionsRow() {
  const mine = useMyProactiveSettings()
  const change = useChangeMyProactiveSettings()
  if (mine.data === undefined && mine.isError) return <FailureLine error={mine.error} />
  if (mine.data === undefined) {
    return (
      <p role="status" className="text-xs text-muted-foreground">
        {LOADING}
      </p>
    )
  }
  return (
    <div className="border-t border-border pt-2">
      <SwitchRow
        label={PAUSE_LABEL}
        detail={PAUSE_LINE}
        checked={mine.data.paused}
        disabled={change.isLoading}
        onCheckedChange={(paused) => change.mutate({ paused })}
      />
      <FailureLine error={change.isError ? change.error : null} />
    </div>
  )
}
