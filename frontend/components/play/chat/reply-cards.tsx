'use client'

import { useMemo } from 'react'

import { TaskCard } from '@/components/chatbot/task-card'
import { WithdrawOfferCard } from '@/components/chatbot/withdraw-offer-card'
import type { ChatMessage } from '@/types'

import { taskCardsOf } from './message-text'
import styles from './reply-cards.module.css'

/**
 * PRD-260 (P260-FX-29, F1) — the core's own cards a reply carries, inside
 * Auto's reply, exactly as Studio's message (components/chatbot/message.tsx)
 * renders them: the withdraw offer (F091-C2: after a "no" in chat, Withdraw
 * denies the request still waiting on a yes through the core's useDenyApproval,
 * and the card then reads "Withdrawn."), and the tickets the reply filed or
 * checked (PRD-238 S6 live, PRD-239 S2 stored), each the core's TaskCard, which
 * follows the board and opens its ticket there. The cards draw on Studio's
 * tokens, which the look writes under Play (D3); Play only frames them.
 */

export const TICKETS_LABEL = 'Tickets'

export interface ReplyCardsProps {
  readonly message: ChatMessage
}

export function ReplyCards({ message }: ReplyCardsProps) {
  const { parts, taskCards, withdrawOffer } = message
  const tickets = useMemo(() => taskCardsOf({ parts, taskCards }), [parts, taskCards])
  if (!withdrawOffer && tickets.length === 0) return null
  return (
    <div className={styles.cards}>
      {withdrawOffer && <WithdrawOfferCard offer={withdrawOffer} />}
      {tickets.length > 0 && (
        <section className={styles.cards} aria-label={TICKETS_LABEL}>
          {tickets.map((card) => (
            <TaskCard key={card.id} card={card} />
          ))}
        </section>
      )}
    </div>
  )
}
