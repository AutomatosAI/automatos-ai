'use client'

/**
 * PRD-277 US-501 — the enterprise edition's answers to lib/auth-hooks.ts's five hooks, filled from
 * the API's who-am-I (session-context.ts). lib/auth-hooks.ts gives them the same types as Clerk's
 * hooks, so a component reads `isSignedIn`, `user.fullName` or `signOut()` whatever the edition.
 *
 * Only the fields the app reads are filled. There is never a token (`getToken` answers null): the
 * session cookie travels by itself. There is never an organisation: workspaces come from the API.
 * Nothing here decides a role; RoleProvider reads the API's `system_role` from the context.
 */
import { useMemo } from 'react'

import type { WhoAmI } from './auth-api'
import { useEnterpriseSession, type EnterpriseSession } from './session-context'

const noToken = async (): Promise<string | null> => null
const nothing = (): void => {}

function sessionOf(session: EnterpriseSession | null) {
  const status = session?.status ?? 'loading'
  return {
    isLoaded: status !== 'loading',
    isSignedIn: status === 'signed-in',
    signOut: session?.signOut ?? (async () => {}),
  }
}

/** The person as Clerk's `user` names the fields the app reads; null before the API answers. */
export function userFrom(who: WhoAmI | null) {
  if (!who) return null
  const { id, email, name } = who.person
  const fullName = name || email || null
  const emailAddress = email ? { emailAddress: email } : null
  return {
    id: String(id),
    fullName,
    firstName: fullName ? fullName.split(' ')[0] : null,
    lastName: null,
    username: null,
    imageUrl: '',
    hasImage: false,
    primaryEmailAddress: emailAddress,
    emailAddresses: emailAddress ? [emailAddress] : [],
    publicMetadata: {},
  }
}

export function useEnterpriseAuth() {
  const session = useEnterpriseSession()
  const { isLoaded, isSignedIn, signOut } = sessionOf(session)
  const userId = session?.who ? String(session.who.person.id) : null
  return useMemo(
    () => ({ isLoaded, isSignedIn, userId, sessionId: null, orgId: null, orgRole: null, orgSlug: null, getToken: noToken, signOut }),
    [isLoaded, isSignedIn, userId, signOut],
  )
}

export function useEnterpriseUser() {
  const session = useEnterpriseSession()
  const { isLoaded, isSignedIn } = sessionOf(session)
  const who = session?.who ?? null
  return useMemo(() => ({ isLoaded, isSignedIn, user: userFrom(who) }), [isLoaded, isSignedIn, who])
}

export function useEnterpriseSessionHook() {
  const { isLoaded, isSignedIn } = sessionOf(useEnterpriseSession())
  return useMemo(() => ({ isLoaded, isSignedIn, session: null }), [isLoaded, isSignedIn])
}

export function useEnterpriseOrganization() {
  const { isLoaded } = sessionOf(useEnterpriseSession())
  return useMemo(() => ({ isLoaded, organization: null, membership: null }), [isLoaded])
}

export function useEnterpriseClerk() {
  const { signOut } = sessionOf(useEnterpriseSession())
  return useMemo(() => ({ signOut, openUserProfile: nothing, openOrganizationProfile: nothing }), [signOut])
}
