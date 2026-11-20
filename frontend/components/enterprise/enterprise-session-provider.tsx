'use client'

/**
 * PRD-277 US-501, US-502 — the enterprise edition's auth boundary (providers.tsx AuthBoundary),
 * in place of ClerkProvider (saas) and LocalAuthProvider (local). No Clerk symbol runs here.
 *
 * - apiClient's token getter answers null: the API's session cookie, which scripts cannot read,
 *   travels with every same-origin request by itself (lib/enterprise/session-transport.ts).
 * - A request made outside apiClient gets the session too: the wrapper over `window.fetch`
 *   (lib/enterprise/session-fetch.ts), put on once here and taken off when this unmounts.
 * - Who the person is and their install role come from `GET /api/v1/auth/me`, never the browser.
 * - The app renders once the API has answered and the workspace is stored. A person in no
 *   workspace sees "ask an administrator to add you"; a 401 sends the browser to sign in. A
 *   public page (lib/public-routes.ts) renders whatever the API answers.
 * - When a later request from the app's own API answers 401, "Your session ended" opens over the page.
 */
import { useEffect, useMemo, useRef, type ReactNode } from 'react'
import { usePathname } from 'next/navigation'

import { apiClient } from '@/lib/api-client'
import { settleKeptDraft } from '@/lib/enterprise/kept-chat-draft'
import {
  LOADING,
  SIGN_OUT,
  TRY_AGAIN,
  UNREACHABLE_BODY,
  UNREACHABLE_TITLE,
  WAITING_BODY,
  WAITING_TITLE,
} from '@/lib/enterprise/messages'
import {
  EnterpriseSessionContext,
  signOutOfEnterprise,
  type EnterpriseSession,
} from '@/lib/enterprise/session-context'
import { installSessionFetch } from '@/lib/enterprise/session-fetch'
import { useSessionState, type SessionState } from '@/lib/enterprise/use-session-state'

import { EnterpriseNotice } from './enterprise-notice'
import { SessionEndedDialog } from './session-ended-dialog'

const signOut = () => signOutOfEnterprise()

/** What renders in place of the app, or null when the app itself renders. */
function noticeFor(state: SessionState): ReactNode {
  if (state.onSignInPage || state.onPublicPage) return null
  if (state.load === 'failed') {
    return <EnterpriseNotice title={UNREACHABLE_TITLE} body={UNREACHABLE_BODY} action={{ label: TRY_AGAIN, onClick: state.reload }} />
  }
  if (state.load !== 'signed-in') return <EnterpriseNotice title={LOADING} />
  if (state.workspaceId === null) {
    return <EnterpriseNotice title={WAITING_TITLE} body={WAITING_BODY} action={{ label: SIGN_OUT, onClick: () => void signOut() }} />
  }
  return null
}

/**
 * The one-time setup, during the first render so it is in place before any child sends a request
 * (as LocalAuthProvider does): apiClient's null token getter and the session fetch. The effect puts
 * the fetch back when this unmounts, and on again after a remount (React's strict mode).
 */
function useOneTimeSetup(): void {
  const configuredRef = useRef(false)
  const restoreRef = useRef<(() => void) | null>(null)
  if (!configuredRef.current) {
    apiClient.setClerkTokenGetter(async () => null)
    if (typeof window !== 'undefined') restoreRef.current = installSessionFetch(window)
    configuredRef.current = true
  }
  useEffect(() => {
    if (restoreRef.current === null) restoreRef.current = installSessionFetch(window)
    settleKeptDraft()
    return () => {
      restoreRef.current?.()
      restoreRef.current = null
    }
  }, [])
}

export function EnterpriseSessionProvider({ children }: { children: ReactNode }) {
  useOneTimeSetup()
  const state = useSessionState(usePathname())
  const session = useMemo<EnterpriseSession>(
    () => ({
      status: state.load === 'signed-in' ? 'signed-in' : state.load === 'loading' ? 'loading' : 'signed-out',
      who: state.who,
      workspaceId: state.workspaceId,
      signOut,
    }),
    [state.load, state.who, state.workspaceId],
  )
  const notice = noticeFor(state)

  return (
    <EnterpriseSessionContext.Provider value={session}>
      {notice ?? children}
      {state.ended && state.load === 'signed-in' && !state.onSignInPage && <SessionEndedDialog reason={state.endedReason} />}
    </EnterpriseSessionContext.Provider>
  )
}
