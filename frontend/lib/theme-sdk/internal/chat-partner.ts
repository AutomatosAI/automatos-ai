/**
 * SDK 1.5.0 — who answers a theme's chat, and Plan: the chat area's `agentId`, `chooseAgent`, `planMode` and
 * `setPlanMode` (HostChatController, chat.ts), as Studio's chat has them (components/chatbot/chat.tsx).
 *
 * - The agent is the conversation's own choice, as Studio's selector is its Chat's: Auto (null) when the
 *   conversation's screen mounts. `chooseAgent(id)` picks one of the roster's agents (the host's useAgents, the cache
 *   Studio's selector reads). A session agent (`configuration.runtime === 'cli'`) is never a chat partner: Studio
 *   opens its Claude Code session instead (handleAgentChange), and here the choice is refused (false), as it is for an
 *   id the roster doesn't have, or while the roster loads. A theme opens a session agent with a link to
 *   `/chat?agent=<id>` (components/chatbot/agent-link.tsx). A value that is neither a whole number nor null throws.
 * - Plan is the host's own switch (stores/mission-store.ts `isPlanMode`), the one Studio's Plan button and a
 *   `/chat?mode=plan` link set, so the theme and Studio always agree. A value other than true or false throws.
 *
 * The controller sends both with the next message (lib/chat-controller/controller.ts), and the AI label names the
 * agent while it answers (lib/chat/use-chat-with-receipts.ts).
 */
import { useCallback, useMemo, useState } from 'react'

import { isSessionAgent, type Agent } from '@/components/chatbot/agent-selector'
import { useAgents } from '@/hooks/use-agent-api'
import { useMissionStore } from '@/stores/mission-store'

import { refuse } from './query-view'

/** Who answers the conversation, and Plan. */
export interface ChatPartner {
  readonly agentId: number | null
  readonly chooseAgent: (id: number | null) => boolean
  readonly planMode: boolean
  readonly setPlanMode: (on: boolean) => void
}

const NOT_AN_AGENT_ID = 'chooseAgent takes an agent id from useAgents (a whole number), or null for Auto.'
const NOT_ON_OR_OFF = 'setPlanMode takes true or false.'

function isAgentId(id: unknown): id is number {
  return typeof id === 'number' && Number.isSafeInteger(id) && id > 0
}

/** The roster's agent with that id, or undefined (not in the roster, or the roster still loading). */
function rosterAgent(roster: unknown, id: number): Agent | undefined {
  return Array.isArray(roster) ? (roster as Agent[]).find((agent) => agent.id === id) : undefined
}

/** The conversation's agent (Auto to start) and the host's Plan switch, for a theme's chat controller. */
export function useChatPartner(): ChatPartner {
  const [agentId, setAgentId] = useState<number | null>(null)
  const { data: roster } = useAgents()
  const planMode = useMissionStore((s) => s.isPlanMode)
  const setHostPlanMode = useMissionStore((s) => s.setPlanMode)
  const chooseAgent = useCallback(
    (id: number | null): boolean => {
      if (id === null) {
        setAgentId(null)
        return true
      }
      if (!isAgentId(id)) return refuse(NOT_AN_AGENT_ID)
      const agent = rosterAgent(roster, id)
      if (agent === undefined || isSessionAgent(agent)) return false
      setAgentId(id)
      return true
    },
    [roster],
  )
  const setPlanMode = useCallback(
    (on: boolean): void => {
      if (typeof on !== 'boolean') refuse(NOT_ON_OR_OFF)
      setHostPlanMode(on)
    },
    [setHostPlanMode],
  )
  return useMemo(() => ({ agentId, chooseAgent, planMode, setPlanMode }), [agentId, chooseAgent, planMode, setPlanMode])
}
