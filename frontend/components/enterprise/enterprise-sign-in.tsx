'use client'

/**
 * PRD-277 US-502 — the enterprise sign-in page: one button, "Sign in with <OIDC_DISPLAY_NAME>",
 * that sends the browser to the API's start route. No password field, no sign-up, no reset: the
 * organisation's identity provider does all of that.
 *
 * `return_to` (where the person was going) is passed through to the start route untouched; the
 * API's rule (`safe_return_path`) decides whether it may be used. `error` (the API's refusal code)
 * picks one sentence from lib/enterprise/messages.ts; an unknown code reads as the general one, so
 * the address bar can never make the page say something the app did not write.
 */
import { Suspense, useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { LogIn } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { DEFAULT_DISPLAY_NAME, RETURN_TO_PARAM, fetchDisplayName, signInStartUrl } from '@/lib/enterprise/auth-api'
import { SIGN_IN_ERROR_PARAM, signInRefusalMessage } from '@/lib/enterprise/messages'

function useDisplayName(): string {
  const [name, setName] = useState(DEFAULT_DISPLAY_NAME)
  useEffect(() => {
    let live = true
    void fetchDisplayName().then((answer) => {
      if (live) setName(answer)
    })
    return () => {
      live = false
    }
  }, [])
  return name
}

function SignInCard() {
  const params = useSearchParams()
  const displayName = useDisplayName()
  const refusal = signInRefusalMessage(params.get(SIGN_IN_ERROR_PARAM))
  const start = signInStartUrl(params.get(RETURN_TO_PARAM))

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground">
      <section
        aria-labelledby="enterprise-sign-in-title"
        className="w-full max-w-sm space-y-5 rounded-2xl border border-border bg-card p-6 text-card-foreground"
      >
        <div className="space-y-2">
          <h1 id="enterprise-sign-in-title" className="text-xl font-semibold">Sign in to Automatos</h1>
          <p className="text-sm text-muted-foreground">Your organisation manages your account.</p>
        </div>
        {refusal !== null && <p role="alert" className="text-sm text-destructive">{refusal}</p>}
        <Button asChild className="w-full">
          <a href={start}>
            <LogIn className="mr-2 h-4 w-4" />
            Sign in with {displayName}
          </a>
        </Button>
      </section>
    </main>
  )
}

export function EnterpriseSignIn() {
  return (
    <Suspense fallback={null}>
      <SignInCard />
    </Suspense>
  )
}
