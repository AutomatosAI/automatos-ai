'use client'

import { AuthenticateWithRedirectCallback } from '@clerk/nextjs'
import { redirect } from 'next/navigation'
import { isRouteAvailableInEdition } from '@/lib/auth-edition'

export default function SSOCallbackPage() {
  // PRD-233 S7: Clerk's SSO return leg — no login in the local edition; send home.
  if (!isRouteAvailableInEdition('/sso-callback')) {
    redirect('/')
  }
  return (
    <AuthenticateWithRedirectCallback
      afterSignInUrl="/"
      afterSignUpUrl="/"
      redirectUrl="/"
    />
  )
}
