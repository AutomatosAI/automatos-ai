'use client'

import { type ReactNode, useEffect, useRef } from 'react'

import { useAgents } from '@/hooks/use-agent-api'
import { AI_LABEL_PAUSED_TEXT, UNNAMED_AGENT, aiLabelText } from '@/lib/ai-label/copy'
import { useAiLabelPaused, useAiLabelStore } from '@/lib/ai-label/store'

import { useAiLabelCheck } from './use-ai-label-check'

/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — the host's AI label on every in-app chat.
 *
 * A route that holds a chat wraps its page in <AiLabelGate> (app/chat/layout.tsx: the default
 * chat with Mission Mode, Play's chat and any theme's /chat; app/conversations/page.tsx). The
 * gate renders the label after the page, outside the route switch and outside the theme's page
 * slot, so no page, frame or theme component draws it or can leave it out. It reads "Auto is an
 * AI", or "{agent name} is an AI" while the owner's picked agent answers, in theme tokens, with
 * role="note" and the same words for screen readers.
 *
 * No theme can hide it: the label checks it can be seen (use-ai-label-check.ts) and publishes
 * the answer (lib/ai-label/store.ts). While it can't be seen, every chat composer's fence
 * (ai-label-fence.tsx) disables the input, and the gate says why (AI_LABEL_PAUSED_TEXT). No
 * prop, manifest field or setting turns the label off.
 */

export const AI_LABEL_ATTR = 'data-ai-label'

/** Where the label sits: a page with Auto's floating widget at the bottom right takes the left. */
export type AiLabelPlacement = 'bottom-right' | 'bottom-left'

const PLACEMENT_CLASS: Readonly<Record<AiLabelPlacement, string>> = {
  'bottom-right': 'right-3',
  'bottom-left': 'left-3',
}
const LABEL_CLASS =
  'pointer-events-none fixed bottom-2 z-40 rounded-full border border-border bg-background px-2.5 py-0.5 ' +
  'text-xs font-medium text-foreground shadow-sm'
const PAUSED_CLASS =
  'fixed bottom-10 z-40 max-w-[calc(100%-1.5rem)] rounded-md border border-destructive bg-background px-3 py-2 ' +
  'text-sm text-foreground shadow-md'

interface AiLabelNoteProps {
  readonly name: string | null
  readonly placement: AiLabelPlacement
  readonly onHiddenChange: (hidden: boolean) => void
}

/** The label's element: checked for visibility from the moment it mounts. */
function AiLabelNote({ name, placement, onHiddenChange }: AiLabelNoteProps) {
  const ref = useRef<HTMLParagraphElement>(null)
  useAiLabelCheck(ref, onHiddenChange)
  const text = aiLabelText(name)
  return (
    <p ref={ref} role="note" aria-label={text} data-ai-label="" className={`${LABEL_CLASS} ${PLACEMENT_CLASS[placement]}`}>
      {text}
    </p>
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
  /** Told true when the label can't be seen, false when it can. */
  readonly onHiddenChange: (hidden: boolean) => void
}

/** "Auto is an AI", or "{agent name} is an AI" while a picked agent answers. */
export function AiLabel({ placement = 'bottom-right', onHiddenChange }: AiLabelProps) {
  const agentId = useAiLabelStore((state) => state.agentId)
  if (agentId === null) return <AiLabelNote name={null} placement={placement} onHiddenChange={onHiddenChange} />
  return <PickedAgentNote agentId={agentId} placement={placement} onHiddenChange={onHiddenChange} />
}

export interface AiLabelGateProps {
  readonly children: ReactNode
  readonly placement?: AiLabelPlacement
}

/** A chat route's page with the host's AI label after it, and the pause notice while the label is hidden. */
export function AiLabelGate({ children, placement = 'bottom-right' }: AiLabelGateProps) {
  const hidden = useAiLabelPaused()
  const setHidden = useAiLabelStore((state) => state.setHidden)
  // A gate that goes takes its pause with it; the next one checks its own label as it mounts.
  useEffect(() => () => setHidden(false), [setHidden])
  return (
    <>
      {children}
      <AiLabel placement={placement} onHiddenChange={setHidden} />
      {hidden ? (
        <p role="alert" className={`${PAUSED_CLASS} ${PLACEMENT_CLASS[placement]}`}>
          {AI_LABEL_PAUSED_TEXT}
        </p>
      ) : null}
    </>
  )
}
