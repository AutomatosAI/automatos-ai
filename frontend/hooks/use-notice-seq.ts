import { useRef } from 'react'

/**
 * PRD-260 P260-FX-24 (M22) — numbers each notice a component tells, so a
 * repeat of the same words is announced again. A live region only speaks when
 * its content changes: the same import twice, Copy JSON twice or the same
 * failure twice leave its text as it was, and a screen reader hears nothing.
 * The owner renders the words in a <span key={notice.seq}> inside its polite
 * region, and keys its role=alert by seq, so every telling is a new node.
 * Used by the Looks page's notices (components/looks/look-notice.tsx) and the
 * style sheet's Copy tokens (components/looks/style/style-screen.tsx).
 */

/** A notice as told: `seq` numbers each telling, counting from 1. */
export type Told<T> = T & { readonly seq: number }

/** Stamps each notice with the next number from a counter kept for the component's life. */
export function useNoticeSeq(): <T extends object>(notice: T) => Told<T> {
  const count = useRef(0)
  return <T extends object>(notice: T): Told<T> => {
    count.current += 1
    return { ...notice, seq: count.current }
  }
}
