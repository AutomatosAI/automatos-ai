/**
 * PRD-261 A-12a — the SDK's chat reply cards: `cards.TaskCard`, `cards.WithdrawOfferCard`,
 * `cards.MissionSuggestionCard` and `cards.MissionCreatedCard`.
 *
 * Each is the host's own card (components/chatbot/*), given the payload the chat controller's view
 * model carries, so its buttons make the host's calls (the board, the approval grants, Missions)
 * and a theme never draws its own version (PRD §4.4). The props are the SDK's own, read-only
 * types; each wrapper hands the host's card a copy it may keep.
 */
import type { ReactElement } from 'react'

import { MissionCreatedCard as HostMissionCreatedCard } from '@/components/chatbot/mission-created-card'
import { MissionSuggestionCard as HostMissionSuggestionCard } from '@/components/chatbot/mission-suggestion-card'
import { TaskCard as HostTaskCard } from '@/components/chatbot/task-card'
import { WithdrawOfferCard as HostWithdrawOfferCard } from '@/components/chatbot/withdraw-offer-card'

/** A board ticket as the reply that filed or checked it carries it. */
export interface TaskCardData {
  readonly id: number
  readonly number?: string | null
  readonly title: string
  readonly status: string
  readonly assigned_agent: string
  readonly runtime?: string | null
  readonly last_tool?: string | null
  readonly files_touched?: number
  readonly exit_reason?: string | null
  readonly denials?: number
  readonly started_at?: string | null
  readonly completed_at?: string | null
}

export interface TaskCardProps {
  readonly card: TaskCardData
}

/** One request of Auto's still waiting on a yes. */
export interface WithdrawRequest {
  readonly grant_id: number
  readonly action?: string | null
  readonly reason?: string | null
}

/** The owner said no while those requests still wait: Withdraw denies each grant. */
export interface WithdrawOffer {
  readonly message: string
  readonly requests: readonly WithdrawRequest[]
}

export interface WithdrawOfferCardProps {
  readonly offer: WithdrawOffer
}

export interface MissionSuggestionMessage {
  readonly role: string
  readonly content: string
}

export interface MissionSuggestionCardProps {
  readonly goal: string
  readonly complexity: string
  readonly agentId?: number | string
  readonly chatId: string
  readonly recentMessages?: readonly MissionSuggestionMessage[]
}

export interface MissionCreatedCardProps {
  readonly missionId: string
}

/** The host's live card for a board ticket. */
export function TaskCard({ card }: TaskCardProps): ReactElement {
  return <HostTaskCard card={card} />
}

/** The host's withdraw-offer card. */
export function WithdrawOfferCard({ offer }: WithdrawOfferCardProps): ReactElement {
  return <HostWithdrawOfferCard offer={{ message: offer.message, requests: [...offer.requests] }} />
}

/** The host's card offering to turn the conversation into a Mission. */
export function MissionSuggestionCard({ recentMessages, ...offer }: MissionSuggestionCardProps): ReactElement {
  return <HostMissionSuggestionCard {...offer} recentMessages={recentMessages === undefined ? undefined : [...recentMessages]} />
}

/** The host's card for the Mission the chat created. */
export function MissionCreatedCard({ missionId }: MissionCreatedCardProps): ReactElement {
  return <HostMissionCreatedCard missionId={missionId} />
}
