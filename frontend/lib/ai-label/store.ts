'use client'

import { useEffect } from 'react'
import { create } from 'zustand'

/**
 * PRD-264 EUAI-11a, EUAI-11b — the host AI labels' shared state: which agent the chat on screen is
 * talking to, which labels are on screen, and which of them a theme hides.
 *
 * There are two labels, one per kind of in-app chat. The page's label sits in the chat route's
 * layout, outside the page and the theme's slot (components/ai-label/ai-label.tsx); Auto's
 * floating widget draws its own inside its panel (components/ai-label/widget-ai-label.tsx). Each
 * label publishes, at every check, whether a theme keeps it from being seen. A hidden label is
 * drawn again in the host's own layer (components/ai-label/protected-caption.tsx), so a mounted
 * label is on screen either way, and no chat is paused for it.
 * The theme SDK's chat (lib/theme-sdk/chat.ts) starts a turn only while the page's label is
 * mounted (EUAI-11b): a theme's chat on a page with no host label sends nothing. A label that
 * unmounts is released: neither shown nor hidden.
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
  /** The mounted labels a theme keeps from being seen: the host draws each in its own layer instead. */
  readonly hiddenLabels: readonly AiLabelId[]
  /** The mounted labels, checked at least once: on screen, in the page or in the host's own layer. */
  readonly shownLabels: readonly AiLabelId[]
  readonly setAgentId: (agentId: number | null) => void
  /** A label's check: the label is on screen; `hidden` says a theme hides it, so the host's layer shows it. */
  readonly setHidden: (label: AiLabelId, hidden: boolean) => void
  /** The label unmounted: it is neither shown nor hidden until a label of that id checks again. */
  readonly releaseLabel: (label: AiLabelId) => void
}

/** `labels` with `label` in it when `included`, without it otherwise (a new list either way). */
function withLabel(labels: readonly AiLabelId[], label: AiLabelId, included: boolean): readonly AiLabelId[] {
  const others = labels.filter((candidate) => candidate !== label)
  return included ? [...others, label] : others
}

export const useAiLabelStore = create<AiLabelState>((set) => ({
  agentId: null,
  hiddenLabels: [],
  shownLabels: [],
  setAgentId: (agentId) => set({ agentId }),
  setHidden: (label, hidden) =>
    set((state) => ({
      hiddenLabels: withLabel(state.hiddenLabels, label, hidden),
      shownLabels: withLabel(state.shownLabels, label, true),
    })),
  releaseLabel: (label) =>
    set((state) => ({
      hiddenLabels: withLabel(state.hiddenLabels, label, false),
      shownLabels: withLabel(state.shownLabels, label, false),
    })),
}))

/** Whether a theme hides `label`, so the host draws it in its own layer. */
export function useAiLabelHidden(label: AiLabelId = PAGE_AI_LABEL): boolean {
  return useAiLabelStore((state) => state.hiddenLabels.includes(label))
}

/** Whether `label` is on screen right now (mounted and checked, in the page or the host's layer): read at call time. */
export function aiLabelShown(label: AiLabelId): boolean {
  return useAiLabelStore.getState().shownLabels.includes(label)
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
