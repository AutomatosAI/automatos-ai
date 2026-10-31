/**
 * PRD-264 EUAI-11b — a theme's chat starts no turn on a page without the host's AI label.
 *
 * The host wraps every route a theme's chat can open on (/chat and every /x/ route) in its AI label
 * (AiLabelGate, components/ai-label/ai-label.tsx), and no theme can hide that label: one a theme keeps
 * from being seen is drawn again in the host's own layer (components/ai-label/protected-caption.tsx).
 * So a mounted label is on screen, and the chat runs. A theme could still open a chat in its frame on a
 * page that has no label at all, so Send and Try again ask the label's store (lib/ai-label/store.ts) at
 * the moment they are called, and refuse (false, nothing sent) unless the page's label is mounted. A
 * Send kept from an earlier render asks again. Stop is never gated: ending a turn needs no label.
 */
import { PAGE_AI_LABEL, aiLabelShown } from '@/lib/ai-label/store'

/** `action`, refused (false, not called) while no page AI label is mounted. */
export function gatedByAiLabel<Args extends unknown[]>(action: (...args: Args) => boolean): (...args: Args) => boolean {
  return (...args: Args) => aiLabelShown(PAGE_AI_LABEL) && action(...args)
}
