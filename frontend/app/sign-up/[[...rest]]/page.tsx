'use client'

import { SignUpForm } from '@/components/auth/sign-up-form'

export default function SignUpPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6 py-12">
      <div className="relative z-10 w-full max-w-md">
        <SignUpForm />
      </div>
    </div>
  )
}
