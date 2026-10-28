'use client'

import { type ReactNode, type RefObject, useCallback, useEffect, useRef } from 'react'

import { aiLabelText } from '@/lib/ai-label/copy'
import { WIDGET_AI_LABEL, useAiLabelHidden, useAiLabelStore } from '@/lib/ai-label/store'

import { type CaptionInset, ProtectedAiCaption, captionPlaceInside } from './protected-caption'
import { useAiLabelCheck } from './use-ai-label-check'

/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — the AI label in Auto's floating widget.
 *
 * MainLayout draws Auto's widget (components/chatbot/chat-widget.tsx) on every page but /chat,
 * pages a theme draws among them, so the page's label (AiLabelGate) is not beside it. The widget's
 * conversation sits in this frame instead: the host's label, "Auto is an AI" (the widget always
 * talks to Auto), first, then the conversation. The label is checked the way the page's is
 * (use-ai-label-check.ts); while a theme keeps it from being seen, the same words are drawn again
 * at the label's spot, the top of the frame, in the host's own layer (protected-caption.tsx), and
 * the widget keeps working: its input and starter chips are never disabled.
 *
 * The panel springs in from opacity 0, so the first check waits WIDGET_LABEL_SETTLE_MS (checked at
 * once, the fade would read as hidden). chat-widget.tsx is over the file-size limit, so it only
 * swaps its conversation's outer element for this frame.
 */

/** Long enough for the panel's spring (framer-motion, stiffness 400, damping 25) to settle. */
export const WIDGET_LABEL_SETTLE_MS = 400

const FRAME_CLASS = 'flex min-h-0 flex-1 flex-col'
const LABEL_CLASS =
  'mx-3 mt-1 self-start rounded-full border border-border bg-background px-2 py-0.5 text-[11px] font-medium text-foreground'
/** Where the label's words sit in the frame (mt-1 and mx-3, plus its border and padding), in px. */
const LABEL_TEXT_INSET: CaptionInset = { top: 7, left: 21 }

interface WidgetAiLabelNoteProps {
  readonly frame: RefObject<HTMLElement>
  /** A theme keeps the label from being seen: its words are drawn in the host's own layer too. */
  readonly hidden: boolean
  readonly onHiddenChange: (hidden: boolean) => void
}

/** The widget's label, checked for visibility once the panel has settled, and its protected copy while hidden. */
function WidgetAiLabelNote({ frame, hidden, onHiddenChange }: WidgetAiLabelNoteProps) {
  const ref = useRef<HTMLParagraphElement>(null)
  useAiLabelCheck(ref, WIDGET_AI_LABEL, onHiddenChange, WIDGET_LABEL_SETTLE_MS)
  const text = aiLabelText(null)
  const place = useCallback(() => captionPlaceInside(frame.current, LABEL_TEXT_INSET, window), [frame])
  return (
    <>
      <p ref={ref} role="note" aria-label={text} data-ai-label="" className={LABEL_CLASS}>
        {text}
      </p>
      {hidden ? <ProtectedAiCaption text={text} place={place} /> : null}
    </>
  )
}

/** The widget's conversation with the host's AI label above it, drawn in the host's own layer while a theme hides it. */
export function WidgetAiLabelFrame({ children }: { readonly children: ReactNode }) {
  const frame = useRef<HTMLDivElement>(null)
  const hidden = useAiLabelHidden(WIDGET_AI_LABEL)
  const setHidden = useAiLabelStore((state) => state.setHidden)
  const releaseLabel = useAiLabelStore((state) => state.releaseLabel)
  const report = useCallback((labelHidden: boolean) => setHidden(WIDGET_AI_LABEL, labelHidden), [setHidden])
  // A panel that closes takes its label with it; the next one checks its own label.
  useEffect(() => () => releaseLabel(WIDGET_AI_LABEL), [releaseLabel])
  return (
    <div ref={frame} className={FRAME_CLASS}>
      <WidgetAiLabelNote frame={frame} hidden={hidden} onHiddenChange={report} />
      {children}
    </div>
  )
}
