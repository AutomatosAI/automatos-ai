'use client'

import { SignIn } from '@clerk/nextjs'
import { redirect } from 'next/navigation'
import { isSaaS } from '@/lib/auth-edition'
import { isEnterprise } from '@/lib/edition-name'
import { EnterpriseSignIn } from '@/components/enterprise/enterprise-sign-in'

export default function SignInPage() {
  // PRD-277 US-502: enterprise signs in at the organisation's identity provider (one button).
  if (isEnterprise) {
    return <EnterpriseSignIn />
  }

  // PRD-175 (F008): there is no login in the `local` edition — send visitors home.
  if (!isSaaS) {
    redirect('/')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6 py-12">
      <div className="relative z-10 w-full max-w-md">
        <SignIn
          appearance={{
            elements: {
              rootBox: 'mx-auto',
              card: 'bg-card border border-border',
            },
          }}
          redirectUrl="/"
        />
      </div>
    </div>
  )
}
