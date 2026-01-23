'use client'

import { SignIn } from '@clerk/nextjs'

export default function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black via-slate-900 to-slate-950 px-6 py-12">
      <SignIn routing="path" path="/auth/signin" />
    </div>
  )
}
