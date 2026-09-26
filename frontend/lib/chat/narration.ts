/**
 * F186 (night 6) — what the model said before calling its tools is narration,
 * not the reply.
 *
 * Night 6 saved 41 of 159 replies as two replies in one: the model's pre-tool
 * guess ("…waiting for the Shopify Support Agent") and then its answer ("This
 * task was completed"). The backend now keeps the answer as the reply and the
 * narration as its own part, and marks each round on the stream as it goes:
 * `d:{"type":"narration","data":{"text": …}}` once a round that called tools has
 * streamed, with `retracted: true` when F108 nudged a claim and its retry
 * replaces it. Live, the text moves out of the answer into the activity trail
 * (a retracted claim is dropped). Reloaded, the stored narration part renders in
 * the trail.
 */
import type { MessagePart } from '@/types'

/** ``content`` without the last occurrence of ``text``: the round just marked. */
export function withoutNarration(content: string, text: string): string {
  if (!text) return content
  const at = content.lastIndexOf(text)
  return at < 0 ? content : content.slice(0, at) + content.slice(at + text.length)
}

/** The stored narration of a saved reply, as activity-trail lines. */
export function narrationLines(parts?: MessagePart[]): string[] {
  return (parts ?? []).flatMap((part) =>
    part.type === 'narration'
      ? part.narration.split(/\n{2,}/).map((line) => line.trim()).filter(Boolean)
      : [],
  )
}
