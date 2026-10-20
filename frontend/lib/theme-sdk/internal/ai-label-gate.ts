/**
 * PRD-264 EUAI-11b — a theme's chat starts no turn unless the host's AI label is on screen.
 *
 * The host wraps every route a theme's chat can open on (/chat and every /x/ route) in its AI label
 * (AiLabelGate, components/ai-label/ai-label.tsx), and a theme's composer sits in cards.AiLabelFence,
 * disabled while that label can't be seen. The fence guards the box, not the call: a theme could
 * still call Send from a button of its own or a timer, or open a chat in its frame on a page that
 * has no label. So Send and Try again ask the label's store (lib/ai-label/store.ts, the state the
 * fence reads) at the moment they are called, and refuse (false, nothing sent) unless the page's
 * label is mounted and was seen at its last check. A Send kept from an earlier render asks again.
 * Stop is never gated: ending a turn needs no label.
 */
import { PAGE_AI_LABEL, aiLabelShown } from '@/lib/ai-label/store'

/** `action`, refused (false, not called) while the page's AI label is not on screen. */
export function gatedByAiLabel<Args extends unknown[]>(action: (...args: Args) => boolean): (...args: Args) => boolean {
  return (...args: Args) => aiLabelShown(PAGE_AI_LABEL) && action(...args)
}
