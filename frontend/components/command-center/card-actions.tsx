/**
 * The decision row of a Command Centre card (a grant, a parked mission, an
 * agent's question): compact buttons at their natural width, right-aligned,
 * wrapping on a narrow screen.
 *
 * #1045: the primary button carried `flex-1`, so Grant, Approve and Answer were
 * card-wide cream bars (Studio's primary is cream in the dark tone) next to
 * small outline buttons. Studio's own action buttons (`.cc-btn`) are 30px,
 * 12px text; these match.
 */

import type { ReactNode } from 'react'

/** Button classes for a card action: pass with `size="sm"`. */
export const CARD_ACTION_BUTTON = 'h-8 px-3 text-xs'

/** The icon inside a card action button. */
export const CARD_ACTION_ICON = 'mr-1 h-3.5 w-3.5'

export function CardActions({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <div
      className="flex flex-wrap items-center justify-end gap-2"
      role={label ? 'group' : undefined}
      aria-label={label}
    >
      {children}
    </div>
  )
}
