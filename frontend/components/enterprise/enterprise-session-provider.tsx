'use client'

/**
 * PRD-277 US-501, US-502 — the enterprise edition's auth boundary (providers.tsx AuthBoundary),
 * in place of ClerkProvider (saas) and LocalAuthProvider (local). No Clerk symbol runs here.
 *
 * - apiClient's token getter answers null: the API's session cookie, which scripts cannot read,
 *   travels with every same-origin request by itself (lib/enterprise/session-transport.ts).
 * - Who the person is and their install role come from `GET /api/v1/auth/me`, never the browser.
 * - The app renders once the API has answered and the workspace is stored. A person in no
 *   workspace sees "ask an administrator to add you"; a 401 sends the browser to sign in.
 * - When a later request meets `session_ended`, "Your session ended" opens over the page.
 */
import { useMemo, useRef, type ReactNode } from 'react'
import { usePathname } from 'next/navigation'

import { apiClient } from '@/lib/api-client'
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
import { useSessionState, type SessionState } from '@/lib/enterprise/use-session-state'

import { EnterpriseNotice } from './enterprise-notice'
import { SessionEndedDialog } from './session-ended-dialog'

const signOut = () => signOutOfEnterprise()

/** What renders in place of the app, or null when the app itself renders. */
function noticeFor(state: SessionState): ReactNode {
  if (state.onSignInPage) return null
  if (state.load === 'failed') {
    return <EnterpriseNotice title={UNREACHABLE_TITLE} body={UNREACHABLE_BODY} action={{ label: TRY_AGAIN, onClick: state.reload }} />
  }
  if (state.load !== 'signed-in') return <EnterpriseNotice title={LOADING} />
  if (state.workspaceId === null) {
    return <EnterpriseNotice title={WAITING_TITLE} body={WAITING_BODY} action={{ label: SIGN_OUT, onClick: () => void signOut() }} />
  }
  return null
}

export function EnterpriseSessionProvider({ children }: { children: ReactNode }) {
  // Installed during render, before any child can send a request (as LocalAuthProvider does).
  const configuredRef = useRef(false)
  if (!configuredRef.current) {
    apiClient.setClerkTokenGetter(async () => null)
    configuredRef.current = true
  }

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
      {state.ended && state.load === 'signed-in' && !state.onSignInPage && <SessionEndedDialog />}
    </EnterpriseSessionContext.Provider>
  )
}
