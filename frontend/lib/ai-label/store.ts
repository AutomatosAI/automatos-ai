'use client'

import { useEffect } from 'react'
import { create } from 'zustand'

/**
 * PRD-264 EUAI-11a — the host AI label's shared state: which agent the chat on screen is talking
 * to, and whether a theme has hidden the label.
 *
 * The label sits in the chat route's layout, outside the page and the theme's slot
 * (components/ai-label/ai-label.tsx), so it can't read the chat's state, and the chat's
 * composers can't read the label's. The chat's stream hook (lib/chat/use-chat-with-receipts.ts)
 * publishes the agent the owner picked; the label says "{agent name} is an AI", or "Auto is an
 * AI" when none is picked. The label publishes whether it can be seen; every composer's fence
 * (components/ai-label/ai-label-fence.tsx) disables its input while it can't.
 */
export interface AiLabelState {
  /** The picked agent's id; null while Auto answers. */
  readonly agentId: number | null
  /** The label is mounted and a theme keeps it from being seen: the chat is paused. */
  readonly hidden: boolean
  readonly setAgentId: (agentId: number | null) => void
  readonly setHidden: (hidden: boolean) => void
}

export const useAiLabelStore = create<AiLabelState>((set) => ({
  agentId: null,
  hidden: false,
  setAgentId: (agentId) => set({ agentId }),
  setHidden: (hidden) => set({ hidden }),
}))

/** Whether the chat is paused because the label can't be seen. */
export function useAiLabelPaused(): boolean {
  return useAiLabelStore((state) => state.hidden)
}

/** Publishes the chat's picked agent while the chat is mounted; Auto again once it unmounts. */
export function usePublishAnsweringAgent(agentId: number | null | undefined): void {
  const setAgentId = useAiLabelStore((state) => state.setAgentId)
  const published = agentId ?? null
  useEffect(() => {
    setAgentId(published)
    return () => setAgentId(null)
  }, [published, setAgentId])
}
