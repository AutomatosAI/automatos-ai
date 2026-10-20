'use client'

import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

import { AI_LABEL_PAUSED_TEXT, aiLabelText } from '@/lib/ai-label/copy'
import { WIDGET_AI_LABEL, useAiLabelPaused, useAiLabelStore } from '@/lib/ai-label/store'

import { AiLabelFence } from './ai-label-fence'
import { useAiLabelCheck } from './use-ai-label-check'

/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — the AI label in Auto's floating widget.
 *
 * MainLayout draws Auto's widget (components/chatbot/chat-widget.tsx) on every page but /chat,
 * pages a theme draws among them, so the page's label (AiLabelGate) is not beside it. The widget's
 * conversation sits in this frame instead: the host's label, "Auto is an AI" (the widget always
 * talks to Auto), first, then the conversation inside the AI label's fence. The label is checked
 * the way the page's is (use-ai-label-check.ts); while a theme keeps it from being seen, the
 * widget's input and starter chips are disabled and the frame says why (AI_LABEL_PAUSED_TEXT).
 * The widget's replies carry no approval cards, so the fence takes the whole conversation.
 *
 * The panel springs in from opacity 0, so the first check waits WIDGET_LABEL_SETTLE_MS, and the
 * fence holds the input until that check has run. chat-widget.tsx is over the file-size limit,
 * so it only swaps its conversation's outer element for this frame.
 */

/** Long enough for the panel's spring (framer-motion, stiffness 400, damping 25) to settle. */
export const WIDGET_LABEL_SETTLE_MS = 400

const FRAME_CLASS = 'flex min-h-0 flex-1 flex-col'
const LABEL_CLASS =
  'mx-3 mt-1 self-start rounded-full border border-border bg-background px-2 py-0.5 text-[11px] font-medium text-foreground'
const PAUSED_CLASS = 'mx-3 mt-1 rounded-md border border-destructive bg-background px-2 py-1 text-xs text-foreground'

/** The widget's label: checked for visibility once the panel has settled. */
function WidgetAiLabelNote({ onHiddenChange }: { readonly onHiddenChange: (hidden: boolean) => void }) {
  const ref = useRef<HTMLParagraphElement>(null)
  useAiLabelCheck(ref, onHiddenChange, WIDGET_LABEL_SETTLE_MS)
  const text = aiLabelText(null)
  return (
    <p ref={ref} role="note" aria-label={text} data-ai-label="" className={LABEL_CLASS}>
      {text}
    </p>
  )
}

/** The widget's conversation with the host's AI label above it, fenced while the label is hidden. */
export function WidgetAiLabelFrame({ children }: { readonly children: ReactNode }) {
  const hidden = useAiLabelPaused(WIDGET_AI_LABEL)
  const setHidden = useAiLabelStore((state) => state.setHidden)
  const [checked, setChecked] = useState(false)
  const report = useCallback(
    (labelHidden: boolean) => {
      setChecked(true)
      setHidden(WIDGET_AI_LABEL, labelHidden)
    },
    [setHidden],
  )
  // A panel that closes takes its pause with it; the next one checks its own label.
  useEffect(() => () => setHidden(WIDGET_AI_LABEL, false), [setHidden])
  return (
    <div className={FRAME_CLASS}>
      <WidgetAiLabelNote onHiddenChange={report} />
      {hidden ? (
        <p role="alert" className={PAUSED_CLASS}>
          {AI_LABEL_PAUSED_TEXT}
        </p>
      ) : null}
      <AiLabelFence label={WIDGET_AI_LABEL} held={!checked}>
        {children}
      </AiLabelFence>
    </div>
  )
}
