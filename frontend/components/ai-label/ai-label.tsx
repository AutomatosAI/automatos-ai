'use client'

import { type ReactNode, useCallback, useEffect, useRef } from 'react'

import { useAgents } from '@/hooks/use-agent-api'
import { UNNAMED_AGENT, aiLabelText } from '@/lib/ai-label/copy'
import { PAGE_AI_LABEL, useAiLabelHidden, useAiLabelStore } from '@/lib/ai-label/store'

import { type CaptionPlace, ProtectedAiCaption } from './protected-caption'
import { useAiLabelCheck } from './use-ai-label-check'

/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — the host's AI label on every in-app chat.
 *
 * A route that holds a chat wraps its page in <AiLabelGate> (app/chat/layout.tsx: the default
 * chat with Mission Mode, Play's chat and any theme's /chat; app/x/[...path]/page.tsx: every theme
 * route, Play's Conversations among them). The gate renders the label after the page, outside the
 * theme's page slot (ThemeSlot) and outside the theme's frame and routes, so no page, frame or theme
 * component draws it or can leave it out. It reads "Auto is an
 * AI", or "{agent name} is an AI" while the owner's picked agent answers: a tiny caption in the
 * theme's muted-foreground token (no border, background or rounding: never a pill), with
 * role="note" and the same words for screen readers.
 *
 * No theme can hide it: the label checks it can be seen (use-ai-label-check.ts) and publishes
 * the answer (lib/ai-label/store.ts). While it can't be seen, the same words are drawn again at the
 * label's spot in the host's own layer, where no theme CSS reaches (protected-caption.tsx), a
 * theme-host warning names what the theme did, and the chat keeps working: nothing is disabled.
 * No prop, manifest field or setting turns the label off. Auto's floating widget, which MainLayout
 * draws on every page but /chat, carries its own label in its panel (widget-ai-label.tsx).
 */

export const AI_LABEL_ATTR = 'data-ai-label'

/** Where the label sits: a page with Auto's floating widget at the bottom right takes the left. */
export type AiLabelPlacement = 'bottom-right' | 'bottom-left'

const PLACEMENT_CLASS: Readonly<Record<AiLabelPlacement, string>> = {
  'bottom-right': 'right-3',
  'bottom-left': 'left-3',
}
/**
 * A tiny muted caption. The padding draws nothing (no border or background); it keeps the words on the
 * spot the label's pill put them, 11 px up from bottom-2 and 23 px in from right-3 or left-3.
 */
const LABEL_CLASS =
  'pointer-events-none fixed bottom-2 z-40 px-[11px] py-[3px] text-[11px] font-medium leading-4 text-muted-foreground'

/** Where the label's words sit (bottom-2 and right-3 or left-3, plus its padding), in px. */
const LABEL_TEXT_BOTTOM_PX = 11
const LABEL_TEXT_SIDE_PX = 23

/** The label's words in the host's own layer go where the label's own words would be. */
const CAPTION_PLACE: Readonly<Record<AiLabelPlacement, () => CaptionPlace>> = {
  'bottom-right': () => ({ bottom: LABEL_TEXT_BOTTOM_PX, right: LABEL_TEXT_SIDE_PX }),
  'bottom-left': () => ({ bottom: LABEL_TEXT_BOTTOM_PX, left: LABEL_TEXT_SIDE_PX }),
}

interface AiLabelNoteProps {
  readonly name: string | null
  readonly placement: AiLabelPlacement
  /** A theme keeps the label from being seen: its words are drawn in the host's own layer too. */
  readonly hidden: boolean
  readonly onHiddenChange: (hidden: boolean) => void
}

/** The label's element, checked for visibility from the moment it mounts, and its protected copy while hidden. */
function AiLabelNote({ name, placement, hidden, onHiddenChange }: AiLabelNoteProps) {
  const ref = useRef<HTMLParagraphElement>(null)
  useAiLabelCheck(ref, PAGE_AI_LABEL, onHiddenChange)
  const text = aiLabelText(name)
  return (
    <>
      <p ref={ref} role="note" aria-label={text} data-ai-label="" className={`${LABEL_CLASS} ${PLACEMENT_CLASS[placement]}`}>
        {text}
      </p>
      {hidden ? <ProtectedAiCaption text={text} place={CAPTION_PLACE[placement]} /> : null}
    </>
  )
}

type NamedAgent = { readonly id?: unknown; readonly name?: unknown }

/** The picked agent's label: its name from the agents list the chat already loaded. */
function PickedAgentNote({ agentId, ...note }: Omit<AiLabelNoteProps, 'name'> & { readonly agentId: number }) {
  const { data } = useAgents()
  const agent = (data as NamedAgent[] | undefined)?.find((candidate) => candidate.id === agentId)
  const name = typeof agent?.name === 'string' && agent.name.trim() ? agent.name : UNNAMED_AGENT
  return <AiLabelNote name={name} {...note} />
}

export interface AiLabelProps {
  readonly placement?: AiLabelPlacement
  /** A theme keeps the label from being seen: its words are drawn in the host's own layer too. */
  readonly hidden: boolean
  /** Told true when the label can't be seen, false when it can. */
  readonly onHiddenChange: (hidden: boolean) => void
}

/** "Auto is an AI", or "{agent name} is an AI" while a picked agent answers. */
export function AiLabel({ placement = 'bottom-right', hidden, onHiddenChange }: AiLabelProps) {
  const agentId = useAiLabelStore((state) => state.agentId)
  const note = { placement, hidden, onHiddenChange }
  if (agentId === null) return <AiLabelNote name={null} {...note} />
  return <PickedAgentNote agentId={agentId} {...note} />
}

export interface AiLabelGateProps {
  readonly children: ReactNode
  readonly placement?: AiLabelPlacement
}

/** A chat route's page with the host's AI label after it, drawn in the host's own layer while a theme hides it. */
export function AiLabelGate({ children, placement = 'bottom-right' }: AiLabelGateProps) {
  const hidden = useAiLabelHidden(PAGE_AI_LABEL)
  const setHidden = useAiLabelStore((state) => state.setHidden)
  const releaseLabel = useAiLabelStore((state) => state.releaseLabel)
  const setPageHidden = useCallback((labelHidden: boolean) => setHidden(PAGE_AI_LABEL, labelHidden), [setHidden])
  // A gate that goes takes its label with it; the next one checks its own label as it mounts.
  useEffect(() => () => releaseLabel(PAGE_AI_LABEL), [releaseLabel])
  return (
    <>
      {children}
      <AiLabel placement={placement} hidden={hidden} onHiddenChange={setPageHidden} />
    </>
  )
}
