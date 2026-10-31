'use client'

/**
 * PRD-270 US-270-09a (§5.7, §8-Q8, K21) — back on the Marketplace item after sign-in or sign-up.
 *
 * A signed-out visitor's Add, Sign in and Start free remember a Marketplace path
 * (lib/marketplace/return-after-sign-in.ts). The sign-in, sign-up and SSO pages stay untouched and
 * land on '/', so this component, mounted once in Providers inside WorkspaceProvider, takes the
 * path once the person is signed in AND has a workspace (a new sign-up's first workspace is made
 * by the existing flow) and router.replace()s to it, once: taking it forgets it. Nothing when no
 * path was remembered, or it expired or is unsafe. SaaS only: the local edition has no sign-in.
 *
 * It never races a page's own navigation. It waits while the person is on a page that navigates
 * once they are in (sign-in, sign-up, the SSO callback, an invitation, a password reset). On the
 * home page, which forwards to Chat from the browser under an installed theme once the theme host
 * has loaded (CoreHomeForward), it waits for HOME_SETTLE_MS: a forward in that time lands it on
 * Chat, where it resumes at once; a theme that draws its own home keeps the person on '/', and it
 * resumes there.
 */
import { useEffect } from 'react'
import { usePathname, useRouter } from 'next/navigation'

import { useWorkspace } from '@/components/workspace-provider'
import { isSaaS } from '@/lib/auth-edition'
import { useAuth } from '@/lib/auth-hooks'
import { take } from '@/lib/marketplace/return-after-sign-in'

/** The pages that navigate on their own once the person is in: the resume waits until they leave. */
const SELF_NAVIGATING_PAGES = /^\/(?:sign-in|sign-up|sso-callback|accept-invitation|reset-password)(?:\/|$)/
const HOME_PATH = '/'
/** How long the home page has to forward on its own before the resume runs there. */
export const HOME_SETTLE_MS = 2000

function currentPath(): string {
  return `${window.location.pathname}${window.location.search}`
}

function Resume() {
  const { isSignedIn } = useAuth()
  const workspaceId = useWorkspace().workspace?.id ?? null
  const pathname = usePathname() ?? ''
  const router = useRouter()
  useEffect(() => {
    if (!isSignedIn || workspaceId === null || SELF_NAVIGATING_PAGES.test(pathname)) return undefined
    const resume = () => {
      const path = take()
      if (path !== null && path !== currentPath()) router.replace(path)
    }
    if (pathname !== HOME_PATH) {
      resume()
      return undefined
    }
    const settle = setTimeout(resume, HOME_SETTLE_MS)
    return () => clearTimeout(settle)
  }, [isSignedIn, workspaceId, pathname, router])
  return null
}

/** Mounted once in Providers, inside WorkspaceProvider; renders nothing. */
export function ResumeAfterSignIn() {
  return isSaaS ? <Resume /> : null
}
