/**
 * PRD-261 A-10a — SDK area `navigation`: moving around the app without importing `next/*`.
 * `useHostRouter` and `useHostPathname` wrap the App Router's own, `HostLink` is the host's
 * `next/link` (an https href opens in a new tab instead), the hrefs are core's own route
 * builders, and `useCoreNav` is the Studio menu after the host's filters.
 *
 * Every href is checked (internal/href.ts): a path inside the app, or https out of it. A theme
 * that hands the router or a link anything else gets a TypeError, never a navigation.
 */
import { forwardRef, useMemo, type AnchorHTMLAttributes, type ReactNode } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useSystemRoleOptional } from '@/contexts/role-context'
import { resolveActiveMenuId } from '@/lib/studio-menu'

import { buildCoreNav, type CoreNav } from './internal/core-nav'
import { assertInternalHref, isExternalHref, isInternalHref } from './internal/href'

export {
  BOARD_HREF,
  CHAT_HREF,
  DELIVERABLES_HREF,
  DELIVERABLE_PARAM,
  LOOKS_HREF,
  LOOK_STYLE_HREF,
  QUESTIONS_HREF,
  chatHref,
  missionHref,
  questionHref,
  ticketHref,
} from '@/lib/ticket-links'
export type { CoreNav, CoreNavItem } from './internal/core-nav'

export interface HostRouter {
  push(href: string): void
  replace(href: string): void
  back(): void
  refresh(): void
  prefetch(href: string): void
}

export interface HostLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href' | 'target' | 'rel'> {
  /** A path inside the app, or an https URL (opened in a new tab). */
  readonly href: string
  readonly children?: ReactNode
}

const EXTERNAL_TARGET = '_blank'
const EXTERNAL_REL = 'noopener noreferrer'

/** The App Router, taking only paths inside the app. */
export function useHostRouter(): HostRouter {
  const router = useRouter()
  return useMemo(() => Object.freeze({
    push: (href: string) => {
      assertInternalHref(href)
      router.push(href)
    },
    replace: (href: string) => {
      assertInternalHref(href)
      router.replace(href)
    },
    back: () => router.back(),
    refresh: () => router.refresh(),
    prefetch: (href: string) => {
      assertInternalHref(href)
      router.prefetch(href)
    },
  }), [router])
}

/** The current path, without the query string. */
export function useHostPathname(): string {
  return usePathname() ?? '/'
}

/** A link: the host's client-side link for a path, a new tab for https, a TypeError for anything else. */
export const HostLink = forwardRef<HTMLAnchorElement, HostLinkProps>(function HostLink({ href, children, ...rest }, ref) {
  if (isInternalHref(href)) {
    return <Link ref={ref} {...rest} href={href}>{children}</Link>
  }
  if (isExternalHref(href)) {
    return <a ref={ref} {...rest} href={href} target={EXTERNAL_TARGET} rel={EXTERNAL_REL}>{children}</a>
  }
  throw new TypeError(`Automatos theme SDK: HostLink cannot link to "${String(href)}".`)
})

/** The core nav for this caller: the Studio menu after the edition, role and plan filters. */
export function useCoreNav(): CoreNav {
  const exposure = useWorkspaceOptional()?.workspace?.exposure
  const isAdmin = useSystemRoleOptional()?.isAdmin === true
  return useMemo(() => buildCoreNav({ isAdmin, exposure }), [isAdmin, exposure])
}

/** The id of the core nav item `pathname` belongs to, or null. */
export function resolveActiveNavId(pathname: string): string | null {
  return resolveActiveMenuId(pathname)
}
