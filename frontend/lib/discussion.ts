/**
 * PRD-252 R2 — Discuss: talk a ticket through with Auto, then "Update ticket and re-queue".
 *
 * Reject sends work back with a note, and after the third the owner and the
 * agent are still talking past each other. Discuss opens a chat with the ticket
 * in the page's context (a reference, PRD-221): Auto reads it, agrees a brief
 * with the owner, and that brief goes back onto the ticket for its agent. On a
 * mission's ticket (D4) Auto discusses the mission; its plan is decided on the
 * mission's page, so nothing is re-queued from the chat.
 */
import type { BoardTask } from '@/types/board'
import type { ChatMessage } from '@/types'

export const CHAT_ROUTE = '/chat'
/** From this many send-backs on, the review panel suggests Discuss (D3). */
export const DISCUSS_AFTER_REJECTS = 3
/** The API's bound on an agreed brief (services/ticket_redo.MAX_BRIEF_CHARS). */
export const MAX_BRIEF_CHARS = 8000

export interface Discussion {
  ticketId: string
  number: string | null
  title: string
  agentName: string | null
  /** Set for a mission's ticket: Auto discusses the mission. */
  missionId: string | null
}

/** `/chat?ticket=<id>`: a chat that discusses the ticket. (With `repo`, the same
 * parameter opens a session ticket's Runtime Canvas instead.) */
export function discussHref(ticketId: string | number): string {
  return `${CHAT_ROUTE}?${new URLSearchParams({ ticket: String(ticketId) }).toString()}`
}

export function discussionOf(task: BoardTask): Discussion {
  return {
    ticketId: String(task.id),
    number: task.number ?? null,
    title: task.name,
    agentName: task.assignee?.agent_name ?? null,
    missionId: task.type === 'mission' && task.mission_id ? task.mission_id : null,
  }
}

/** What the chat's page context selects while the discussion is open: a ticket by its board
 * number (#0042), the number Auto's tools take; a bare id would be read as a number. */
export function discussionSelection(discussion: Discussion): { type: string; id: string } {
  return discussion.missionId
    ? { type: 'mission', id: discussion.missionId }
    : { type: 'board_task', id: discussion.number ?? discussion.ticketId }
}

/** "ticket #0042", or "ticket 612" for one with no number. */
export function discussionLabel(discussion: Pick<Discussion, 'number' | 'ticketId'>): string {
  return `ticket ${discussion.number ?? discussion.ticketId}`
}

/** An installed theme's own chat on screen (components/chatbot/discussion-bar.tsx), by the theme's
 * name (its theme.json `name`, from the active theme's registry entry); null when there is none to show. */
export interface ThemeChatOnScreen {
  readonly themeName: string | null
}

/** Where the hint says the chat is when the theme's name is unknown. */
const UNNAMED_THEME_CHAT = "this theme's chat"

/** "Play's chat" under Play, "HUD's chat" under HUD; the neutral words without a name. */
function themeChatPlace(themeName: string | null): string {
  const name = themeName?.trim()
  return name ? `${name}'s chat` : UNNAMED_THEME_CHAT
}

/** The line under the bar on an agent's ticket: the chat answers as Auto, so it says how to
 * ask the ticket's agent instead. `themeChat`: an installed theme's own chat is on screen, or null
 * for the core's. A theme's message box has no agent picker (installed Play's, PRD-260 M67; a
 * picker there is decision F3), so the way there is Studio. The words are M67's, naming the active
 * theme's chat. Null on a mission's ticket or with no agent. */
export function discussionHint(
  discussion: Pick<Discussion, 'agentName' | 'missionId'>,
  themeChat: ThemeChatOnScreen | null,
): string | null {
  const agent = discussion.agentName
  if (!agent || discussion.missionId) return null
  if (!themeChat) return `You are talking to Auto. To ask ${agent} instead, pick them in the message box.`
  const place = themeChatPlace(themeChat.themeName)
  return `You are talking to Auto in ${place}. To ask ${agent}, switch to Studio from the theme menu.`
}

// The brief Auto proposes comes in one fenced block (services/page_context.py).
const FENCED_BLOCK = /```[^\n]*\n([\s\S]*?)```/

/** The brief Auto proposed in its last reply: its fenced block, else the whole reply. */
export function proposedBrief(messages: ChatMessage[]): string {
  const reply = [...messages].reverse().find((m) => m.role === 'assistant')
  const text = (reply?.parts ?? [])
    .map((part) => (part.type === 'text' && 'text' in part ? part.text : ''))
    .join('\n')
    .trim()
  const block = FENCED_BLOCK.exec(text)
  return (block ? block[1] : text).trim().slice(0, MAX_BRIEF_CHARS)
}
