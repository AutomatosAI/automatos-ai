'use client'

import { useEffect } from 'react'
import { create } from 'zustand'

/**
 * PRD-264 EUAI-11a — the host AI labels' shared state: which agent the chat on screen is talking
 * to, and which labels a theme has hidden.
 *
 * There are two labels, one per kind of in-app chat. The page's label sits in the chat route's
 * layout, outside the page and the theme's slot (components/ai-label/ai-label.tsx); Auto's
 * floating widget draws its own inside its panel (components/ai-label/widget-ai-label.tsx). Each
 * label publishes whether it can be seen, and each composer's fence
 * (components/ai-label/ai-label-fence.tsx) disables its input while the label beside it can't.
 * The chat's stream hook (lib/chat/use-chat-with-receipts.ts) publishes the agent the owner
 * picked; the page's label says "{agent name} is an AI", or "Auto is an AI" when none is picked.
 */

/** The page's label (AiLabelGate) and Auto's floating widget's label. */
export type AiLabelId = 'page' | 'widget'

export const PAGE_AI_LABEL: AiLabelId = 'page'
export const WIDGET_AI_LABEL: AiLabelId = 'widget'

export interface AiLabelState {
  /** The picked agent's id; null while Auto answers. */
  readonly agentId: number | null
  /** The labels that are mounted and that a theme keeps from being seen: their chats are paused. */
  readonly hiddenLabels: readonly AiLabelId[]
  readonly setAgentId: (agentId: number | null) => void
  readonly setHidden: (label: AiLabelId, hidden: boolean) => void
}

/** `labels` with `label` in it when `hidden`, without it otherwise (a new list either way). */
function withHidden(labels: readonly AiLabelId[], label: AiLabelId, hidden: boolean): readonly AiLabelId[] {
  const others = labels.filter((candidate) => candidate !== label)
  return hidden ? [...others, label] : others
}

export const useAiLabelStore = create<AiLabelState>((set) => ({
  agentId: null,
  hiddenLabels: [],
  setAgentId: (agentId) => set({ agentId }),
  setHidden: (label, hidden) => set((state) => ({ hiddenLabels: withHidden(state.hiddenLabels, label, hidden) })),
}))

/** Whether the chat beside `label` is paused because that label can't be seen. */
export function useAiLabelPaused(label: AiLabelId = PAGE_AI_LABEL): boolean {
  return useAiLabelStore((state) => state.hiddenLabels.includes(label))
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
