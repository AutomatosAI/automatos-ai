'use client'

import { MissionApprovalWidget } from '@/components/widgets/MissionApprovalWidget'
import { ToolApprovalWidget } from '@/components/widgets/ToolApprovalWidget'
import type { WidgetMetadata } from '@/components/widgets/types'
import { type FiledApproval, isWaitingCard, MISSION_APPROVAL_TITLE, sourceName, TOOL_APPROVAL_TITLE } from '@/lib/chat/approvals'

import styles from './approval-cards.module.css'

/**
 * PRD-260 T-005c — the approvals a reply raised, inside Auto's reply: the
 * core's own ToolApprovalWidget and MissionApprovalWidget, given the data,
 * title and metadata chat.tsx gives them, so Approve & run, Deny and Reject are
 * the core's calls (useGrantApproval / useDenyApproval, useApproveMission /
 * useRejectMission) exactly as in Studio. The widgets draw on Studio's tokens,
 * which the look writes under Play (D3); Play only frames them. A tool card
 * tells the chat once it is answered (it then shows its outcome); the Mission
 * card closes itself after a decision, as it leaves Studio's canvas. A card
 * whose reply left the conversation (Try again, a request failing mid-turn,
 * the conversation shown again; a reply you stopped stays, its cards in it)
 * shows after it until it is answered or closed: under "Approvals still
 * open" while it waits, under "Approvals" once answered.
 */

export const APPROVALS_LABEL = 'Approvals'
/** The cards whose reply is no longer in the conversation and that still wait for the owner. */
export const LOOSE_APPROVALS_LABEL = 'Approvals still open'

interface CardProps {
  readonly card: FiledApproval
  /** The conversation the card came from ('' before the backend named a new chat). */
  readonly chatId: string
  readonly onAnswered: (key: string) => void
  readonly onClosed: (key: string) => void
}

function metadataOf(card: FiledApproval, chatId: string): WidgetMetadata {
  return {
    source: { type: 'tool', name: sourceName(card) },
    createdAt: card.filedAt,
    conversationId: chatId || undefined,
  }
}

function ApprovalCard({ card, chatId, onAnswered, onClosed }: CardProps) {
  const metadata = metadataOf(card, chatId)
  const close = () => onClosed(card.key)
  if (card.kind === 'mission') {
    return <MissionApprovalWidget id={card.key} title={MISSION_APPROVAL_TITLE} data={card.data} metadata={metadata} onClose={close} />
  }
  return (
    <ToolApprovalWidget
      id={card.key}
      title={TOOL_APPROVAL_TITLE}
      data={card.data}
      metadata={metadata}
      onDecided={() => onAnswered(card.key)}
      onClose={close}
    />
  )
}

export interface ApprovalCardsProps extends Omit<CardProps, 'card'> {
  readonly cards: readonly FiledApproval[]
}

/** The cards a reply raised, inside that reply. */
export function ApprovalCards({ cards, ...actions }: ApprovalCardsProps) {
  if (cards.length === 0) return null
  return (
    <section className={styles.approvals} aria-label={APPROVALS_LABEL}>
      {cards.map((card) => (
        <div key={card.key} className={styles.card}>
          <ApprovalCard card={card} {...actions} />
        </div>
      ))}
    </section>
  )
}

/**
 * The cards whose reply left the conversation, after it (M77): one waiting is
 * under LOOSE_APPROVALS_LABEL, one answered under APPROVALS_LABEL. Each card
 * is its own section, so answering renames the section around the card in
 * place: a card moved into another section would mount afresh, and the tool
 * card would drop the outcome it shows and offer Approve & run again.
 */
export function LooseApprovalCards({ cards, ...actions }: ApprovalCardsProps) {
  if (cards.length === 0) return null
  return (
    <div className={styles.approvals}>
      {cards.map((card) => (
        <section
          key={card.key}
          className={styles.card}
          aria-label={isWaitingCard(card) ? LOOSE_APPROVALS_LABEL : APPROVALS_LABEL}
        >
          <ApprovalCard card={card} {...actions} />
        </section>
      ))}
    </div>
  )
}
