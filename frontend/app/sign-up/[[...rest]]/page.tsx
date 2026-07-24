'use client'

import { redirect } from 'next/navigation'
import { SignUpForm } from '@/components/auth/sign-up-form'
import { isSaaS } from '@/lib/auth-edition'

export default function SignUpPage() {
  // PRD-175 (F008): there is no signup in the `local` edition — send visitors home.
  if (!isSaaS) {
    redirect('/')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6 py-12">
      <div className="relative z-10 w-full max-w-md">
        <SignUpForm />
      </div>
    </div>
  )
}
