'use client'

import type { ReactNode } from 'react'

import { useAiLabelPaused } from '@/lib/ai-label/store'

/**
 * PRD-264 EUAI-11a — a chat composer's fence: while a theme keeps the host's AI label from being
 * seen (lib/ai-label/store.ts), every control of the composer inside it is disabled through
 * <fieldset disabled>, so nobody chats with an AI the page no longer says is one.
 *
 * It wraps the composer only (the default chat's input, Mission Mode's with it, and Play's
 * message box), never a whole chat screen: the approval cards in a reply are protected surfaces
 * a theme may never switch off (docs/theme-host/css-layer-model.md), and they stay clickable.
 * The fieldset is `display: contents`, so it draws no box and changes no layout.
 */

export const AI_LABEL_FENCE_ATTR = 'data-ai-label-fence'

/** No box of its own (Tailwind `contents`), and none of a fieldset's default border, margin or padding. */
const FENCE_CLASS = 'contents m-0 min-w-0 border-0 p-0'

export function AiLabelFence({ children }: { readonly children: ReactNode }) {
  const paused = useAiLabelPaused()
  return (
    <fieldset disabled={paused} role="presentation" data-ai-label-fence="" className={FENCE_CLASS}>
      {children}
    </fieldset>
  )
}
