'use client'

/**
 * OversightNote — PRD-181 S5 (EU-AI-Act Art.14): the risk tier, the risk class
 * and why a human is in the loop, shown on every approval card (the chat's
 * mission and tool approvals, the Governance inbox, a ticket's approvals).
 *
 * One presentation for all of them (#1045): the three copies hardcoded
 * amber-50 with `dark:` variants that never apply, so the note read as a pale
 * yellow slab with 10px amber text on the dark page. It is the shared Alert's
 * compact warning tone now, on theme tokens.
 */

import { ShieldAlert } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

/** Human-readable label for an autonomy oversight tier. */
export function oversightTierLabel(tier?: string): string {
  switch (tier) {
    case 'monitor':
      return 'Monitored'
    case 'human_on_the_loop':
      return 'Human on the loop'
    case 'human_in_the_loop':
      return 'Human approval required'
    default:
      return 'Human approval required'
  }
}

export function OversightNote({
  tier,
  riskClass,
  rationale,
}: {
  tier?: string
  riskClass?: string | null
  rationale?: string | null
}) {
  return (
    <Alert variant="warning" size="compact" role="note" aria-label="Human oversight">
      <ShieldAlert />
      <AlertDescription className="text-xs">
        <p className="font-medium">
          {oversightTierLabel(tier)}
          {riskClass ? ` · ${riskClass.replace(/_/g, ' ')}` : ''}
        </p>
        {rationale && <p className="mt-0.5 text-muted-foreground">{rationale}</p>}
      </AlertDescription>
    </Alert>
  )
}
