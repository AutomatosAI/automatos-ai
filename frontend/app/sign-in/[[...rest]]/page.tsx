'use client'

import { SignInForm } from '@/components/auth/sign-in-form'

export default function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black via-zinc-900 to-black px-6 py-12 relative overflow-hidden">
      {/* Background Element */}
      <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full bg-primary/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] rounded-full bg-orange-600/10 blur-[120px] pointer-events-none" />

      <div className="relative z-10 w-full max-w-md">
        <SignInForm />
      </div>
    </div>
  )
}

