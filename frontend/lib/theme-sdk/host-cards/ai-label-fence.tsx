/**
 * PRD-261 W-A5 (seam owner), PRD-264 EUAI-11a — the SDK's `cards.AiLabelFence`: the host's AI label
 * fence (components/ai-label/ai-label-fence.tsx) around a theme's chat composer.
 *
 * The host draws the AI label beside every chat, and no theme can hide it: while a theme keeps it
 * from being seen, the fence disables every control inside it (a <fieldset disabled> that draws no
 * box), so nobody chats with an AI the page no longer says is one. A theme wraps its composer only,
 * never a whole chat screen: the approval cards in a reply are protected surfaces and stay
 * clickable. This is the host's fence itself, handed the same props the host's own composers give
 * it; the label check and the pause stay the host's.
 */
import type { ReactElement, ReactNode } from 'react'

import { AiLabelFence as HostAiLabelFence } from '@/components/ai-label/ai-label-fence'
import type { AiLabelId } from '@/lib/ai-label/store'

export type { AiLabelId }

export interface AiLabelFenceProps {
  readonly children: ReactNode
  /** The label beside this chat: the page's (the default) or Auto's floating widget's. */
  readonly label?: AiLabelId
  /** Keep the composer disabled until that label has been checked once. */
  readonly held?: boolean
}

/** The host's AI label fence around a theme's composer: disabled while the label beside it can't be seen. */
export function AiLabelFence({ children, label, held }: AiLabelFenceProps): ReactElement {
  return (
    <HostAiLabelFence label={label} held={held}>
      {children}
    </HostAiLabelFence>
  )
}
