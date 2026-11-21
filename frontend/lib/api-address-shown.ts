/**
 * PRD-277 US-505 — the API's address as a person sees it on a page and copies into another system
 * (the Playbook webhook address, components/workflows/playbook-schedule-config.tsx).
 *
 *   enterprise    → the page's own origin: the API answers on the web app's host (PRD-277 §5.3),
 *                   and the build carries no API address (enterprise-build.js). With no page, as on
 *                   the server, '' (a relative address).
 *   local, saas   → exactly as before: the build's NEXT_PUBLIC_API_URL, else http://localhost:8000.
 */
import { isEnterprise } from '@/lib/edition-name'

/** The address the local edition's API answers on by default (docker-compose.yml). */
const LOCAL_API_ADDRESS = 'http://localhost:8000'

/** The part of the page's address this reads (`window.location`). */
export interface PageOrigin {
  readonly origin: string
}

function currentPage(): PageOrigin | null {
  return typeof window === 'undefined' ? null : window.location
}

/** The API's address for a person to copy: the page's origin in enterprise, today's value in the other editions. */
export function apiAddressShown(enterprise: boolean = isEnterprise, page: PageOrigin | null = currentPage()): string {
  if (enterprise) return page?.origin ?? ''
  return process.env.NEXT_PUBLIC_API_URL || LOCAL_API_ADDRESS
}
