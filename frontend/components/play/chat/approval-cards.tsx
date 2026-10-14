'use client'

import { MissionApprovalWidget } from '@/components/widgets/MissionApprovalWidget'
import { ToolApprovalWidget } from '@/components/widgets/ToolApprovalWidget'
import type { WidgetMetadata } from '@/components/widgets/types'

import { type FiledApproval, MISSION_APPROVAL_TITLE, sourceName, TOOL_APPROVAL_TITLE } from './approvals'
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
 * whose reply left the conversation (Stop, Try again) shows after it, under
 * its own label, until it is answered or closed.
 */

export const APPROVALS_LABEL = 'Approvals'
/** The cards whose reply is no longer in the conversation. */
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
  /** The section's name: APPROVALS_LABEL inside a reply, LOOSE_APPROVALS_LABEL after the conversation. */
  readonly label?: string
}

export function ApprovalCards({ cards, label = APPROVALS_LABEL, ...actions }: ApprovalCardsProps) {
  if (cards.length === 0) return null
  return (
    <section className={styles.approvals} aria-label={label}>
      {cards.map((card) => (
        <div key={card.key} className={styles.card}>
          <ApprovalCard card={card} {...actions} />
        </div>
      ))}
    </section>
  )
}
