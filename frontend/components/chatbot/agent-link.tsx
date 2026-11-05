'use client'

/**
 * SDK 1.5.0 — `/chat?agent=<id>`: the chat opens with that agent picked, as if the person had picked it in the
 * message box. A session agent (`configuration.runtime === 'cli'`) is never a chat partner: picking it opens its
 * Claude Code session in the Runtime Canvas (PRD-239 S7 v2). A theme's chat can't open that canvas, so this link is
 * how a theme reaches a session agent (`HostLink href="/chat?agent=<id>"`); `agent` keeps the core chat page under a
 * theme that owns /chat (app/chat/layout.tsx).
 *
 * The pick is Studio's own (chat.tsx handleAgentChange, which the composer is handed as `onAgentChange`), so the
 * session canvas, the selector and the AI label behave exactly as for a pick in the box. It is made once per link:
 * AgentLinkProvider (the chat layout, alive for the visit to /chat) remembers the link it applied, so the composer
 * mounting again (the welcome screen giving way to the conversation, the canvas opening) never picks it again over
 * the person's own choice. Leaving /chat, or a link without `agent`, forgets it.
 *
 * AgentFromLink, drawn by the composer (components/proactive/composer-with-suggestion.tsx) only while a link waits,
 * picks once the conversation on screen has settled (the session hydrated, a cold start landed: the composer's chat
 * is the session's own), the workspace is known (the session canvas needs it) and the roster has the agent (Studio
 * tells a session agent by it). An id the roster doesn't have is never picked.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useSearchParams } from 'next/navigation'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useAgents } from '@/hooks/use-agent-api'
import { useChatSessionStore } from '@/stores/chat-session-store'

/** The query parameter: `/chat?agent=<id>`. */
export const AGENT_LINK_PARAM = 'agent'
const AGENT_ID = /^[1-9]\d{0,9}$/

/** A link's agent waiting to be picked, and how to say it has been. */
export interface PendingAgentLink {
  readonly agentId: number
  readonly take: () => void
}

const AgentLinkContext = createContext<PendingAgentLink | null>(null)

/** The agent id a link names: a positive whole number, else null. */
export function agentIdFromLink(params: Pick<URLSearchParams, 'get'> | null): number | null {
  const value = params?.get(AGENT_LINK_PARAM)?.trim() ?? ''
  return AGENT_ID.test(value) ? Number(value) : null
}

/** Holds the link's agent until it has been picked, for the visit to /chat. */
export function AgentLinkProvider({ children }: { readonly children: ReactNode }) {
  const linked = agentIdFromLink(useSearchParams())
  const [taken, setTaken] = useState<number | null>(null)
  useEffect(() => {
    if (linked === null) setTaken(null)
  }, [linked])
  const take = useCallback(() => setTaken(linked), [linked])
  const pending = useMemo(
    () => (linked !== null && linked !== taken ? { agentId: linked, take } : null),
    [linked, taken, take],
  )
  return <AgentLinkContext.Provider value={pending}>{children}</AgentLinkContext.Provider>
}

/** The link's agent still waiting to be picked; null with none (or outside /chat). */
export function usePendingAgentLink(): PendingAgentLink | null {
  return useContext(AgentLinkContext)
}

/** The composer's conversation is the one the session shows, and a conversation or a new chat is open. */
function useConversationSettled(chatId: string): boolean {
  return useChatSessionStore((s) => {
    const active = s.session.activeChatId
    return s.hydrated && (active ?? '') === chatId && (active !== null || s.session.draftOpen)
  })
}

export interface AgentFromLinkProps {
  readonly link: PendingAgentLink
  /** The composer's conversation ('' for a new chat). */
  readonly chatId: string
  /** The chat's own pick (chat.tsx handleAgentChange). */
  readonly onAgentChange: (agentId: number | null) => void
}

/** Picks the link's agent through the chat's own pick, once everything it needs is known; draws nothing. */
export function AgentFromLink({ link, chatId, onAgentChange }: AgentFromLinkProps): null {
  const { data: roster } = useAgents()
  const workspaceId = useWorkspaceOptional()?.workspace?.id ?? null
  const settled = useConversationSettled(chatId)
  const { agentId, take } = link
  const known = Array.isArray(roster) && roster.some((agent: { id?: unknown }) => agent.id === agentId)
  useEffect(() => {
    if (!known || !settled || workspaceId === null) return
    take()
    onAgentChange(agentId)
  }, [known, settled, workspaceId, agentId, take, onAgentChange])
  return null
}
