'use client'

import { usePathname } from 'next/navigation'
import { resolvePageKey, type PageKey } from '@/lib/generated/page-manifest'

// PRD-221 S5 — structured page context sent with each chat message.
// References only (route/tab/selected-id/filters/visible-ids); Auto fetches
// detail through platform tools. Never carries who the user is — the server
// derives roles itself.
export interface PageContext {
  page: PageKey | 'unknown'
  route: string
  tab?: string
  selected?: { type: string; id: string }
  filters?: Record<string, string>
  visible_ids?: string[]
}

/**
 * Build the current page's context from the route, plus any page-owned extras
 * (a page can pass its active tab, the selected entity, filters, or the ids it
 * is showing). Unmapped routes resolve to the key "unknown" — the backend then
 * renders the minimal one-line context.
 */
export function usePageContext(extra?: Omit<Partial<PageContext>, 'page' | 'route'>): PageContext {
  const pathname = usePathname() || '/'
  return {
    page: resolvePageKey(pathname),
    route: pathname,
    ...extra,
  }
}
