/**
 * PRD-261 W-A5 (seam owner), PRD-264 EUAI-11a — the SDK's `cards.AiLabelFence`: once the host's AI label
 * fence around a theme's chat composer, now its children and nothing else.
 *
 * The host draws the AI label beside every chat, and no theme can hide it: a label a theme keeps from
 * being seen is drawn again in the host's own layer (components/ai-label/protected-caption.tsx), so no
 * composer ever waits on it, and this disables nothing and draws no element of its own. It stays in
 * the SDK because installed themes (Play, HUD) wrap their composers in it and import it by name; a
 * theme may stop using it. `label` and `held` are taken and change nothing.
 */
import type { ReactElement, ReactNode } from 'react'

import type { AiLabelId } from '@/lib/ai-label/store'

export type { AiLabelId }

export interface AiLabelFenceProps {
  readonly children: ReactNode
  /** The label beside this chat, the page's or Auto's floating widget's: taken, changes nothing. */
  readonly label?: AiLabelId
  /** Taken, changes nothing: no composer waits on the host's AI label. */
  readonly held?: boolean
}

/** A theme's composer as it is: the host's AI label is never hidden, so nothing here is disabled. */
export function AiLabelFence({ children }: AiLabelFenceProps): ReactElement {
  return <>{children}</>
}
