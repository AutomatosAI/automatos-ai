'use client'

import {
  type HostChatApproval,
  MissionApproval,
  type MissionApprovalData,
  ToolApproval,
  type ToolApprovalData,
} from '@automatos/theme-sdk'

import styles from './approval-cards.module.css'

/**
 * PRD-260 T-005c — the approvals a reply raised, inside Auto's reply: the
 * core's own ToolApprovalWidget and MissionApprovalWidget, given the data,
 * title and metadata chat.tsx gives them, so Approve & run, Deny and Reject are
 * the core's calls (useGrantApproval / useDenyApproval, useApproveMission /
 * useRejectMission) exactly as in Studio. Play places them through the SDK
 * (PRD-261 A-14a): cards.ToolApproval and cards.MissionApproval are those
 * widgets, protected surfaces, with chat.tsx's titles and badge, and each card
 * comes on useChatController's view model with its outcomes bound. The widgets
 * draw on Studio's tokens, which the look writes under Play (D3); Play only
 * frames them. A tool card tells the chat once it is answered (it then shows
 * its outcome); the Mission card closes itself after a decision, as it leaves
 * Studio's canvas. A card whose reply left the conversation (Try again, a
 * request failing mid-turn, the conversation shown again; a reply you stopped
 * stays, its cards in it) shows after it until it is answered or closed: under
 * "Approvals still open" while it waits, under "Approvals" once answered.
 */

export const APPROVALS_LABEL = 'Approvals'
/** The cards whose reply is no longer in the conversation and that still wait for the owner. */
export const LOOSE_APPROVALS_LABEL = 'Approvals still open'

function ApprovalCard({ approval }: { readonly approval: HostChatApproval }) {
  // Where the card came from: its conversation and when it was filed; the badge is the host's (chat.tsx's).
  const origin = { conversationId: approval.chatId || undefined, createdAt: approval.filedAt }
  if (approval.kind === 'mission') {
    return <MissionApproval data={approval.data as MissionApprovalData} onClose={approval.closed} {...origin} />
  }
  return (
    <ToolApproval
      data={approval.data as ToolApprovalData}
      onDecided={() => approval.answered()}
      onClose={approval.closed}
      {...origin}
    />
  )
}

export interface ApprovalCardsProps {
  readonly approvals: readonly HostChatApproval[]
}

/** The cards a reply raised, inside that reply. */
export function ApprovalCards({ approvals }: ApprovalCardsProps) {
  if (approvals.length === 0) return null
  return (
    <section className={styles.approvals} aria-label={APPROVALS_LABEL}>
      {approvals.map((approval) => (
        <div key={approval.key} className={styles.card}>
          <ApprovalCard approval={approval} />
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
export function LooseApprovalCards({ approvals }: ApprovalCardsProps) {
  if (approvals.length === 0) return null
  return (
    <div className={styles.approvals}>
      {approvals.map((approval) => (
        <section
          key={approval.key}
          className={styles.card}
          aria-label={approval.status === 'waiting' ? LOOSE_APPROVALS_LABEL : APPROVALS_LABEL}
        >
          <ApprovalCard approval={approval} />
        </section>
      ))}
    </div>
  )
}
