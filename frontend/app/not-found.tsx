import type { Metadata } from 'next'

import { NotFoundView } from '@/components/play/edges/not-found-view'

/**
 * PRD-260 §4 — the 404, for any address the app doesn't have. A server
 * component, so it can keep the document title Next's default 404 sets
 * (P260-FX-21, M68): every style's tab and screen reader say it, never the
 * app's own title. The page itself is the client NotFoundView, which picks
 * Play's lost screen or the plain page by style and width.
 */

/** Next's own 404 title. */
export const NOT_FOUND_TITLE = '404: This page could not be found.'

export const metadata: Metadata = { title: NOT_FOUND_TITLE }

export default function NotFound() {
  return <NotFoundView />
}
