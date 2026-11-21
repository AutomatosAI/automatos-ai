'use client'

/**
 * PRD-277 US-603 — the page a phone's sign-in waits on, in the enterprise sign-in page's layout.
 *
 * It reads what waits (lib/enterprise/phone-sign-in.ts) and says who is signing in, to which app, on
 * which phone, and that the person should continue only if they started this from the app. Two buttons:
 * Continue (the only step that makes the app's one-time code; the page then opens the app link the API
 * answered) and Cancel. A refusal the API sent here (`?error=`) is one sentence and no button; so is
 * nothing waiting (no cookie, expired, already used or cancelled).
 */
import { Suspense, useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'

import { Button } from '@/components/ui/button'
import {
  PHONE_CANCEL,
  PHONE_CANCELLED,
  PHONE_CONTINUE,
  PHONE_ERROR_PARAM,
  PHONE_FAILED,
  PHONE_NOTHING_WAITING,
  PHONE_ONLY_IF_YOU_STARTED,
  PHONE_REMOVED_BEFORE,
  PHONE_TITLE,
  NOTHING_WAITING_CODE,
  type PhoneWaiting,
  cancelPhoneSignIn,
  continuePhoneSignIn,
  fetchPhoneWaiting,
  phoneOpening,
  phoneRefusalMessage,
  phoneSigningInAs,
  refusalCodeOf,
} from '@/lib/enterprise/phone-sign-in'
import { LOADING } from '@/lib/enterprise/messages'

type Shown =
  | { readonly kind: 'loading' }
  | { readonly kind: 'waiting'; readonly waiting: PhoneWaiting; readonly busy: boolean; readonly failed: boolean }
  | { readonly kind: 'said'; readonly sentence: string; readonly alert: boolean }

const LOADING_STATE: Shown = { kind: 'loading' }

function said(sentence: string, alert = false): Shown {
  return { kind: 'said', sentence, alert }
}

function ready(waiting: PhoneWaiting, busy = false, failed = false): Shown {
  return { kind: 'waiting', waiting, busy, failed }
}

/** What a refused Continue or Cancel turns the page into: nothing waiting, a refusal, or try again. */
function afterRefusal(error: unknown, waiting: PhoneWaiting): Shown {
  const code = refusalCodeOf(error)
  if (code === NOTHING_WAITING_CODE) return said(PHONE_NOTHING_WAITING)
  const refusal = code === null ? null : phoneRefusalMessage(code)
  if (refusal !== null && refusal !== PHONE_FAILED) return said(refusal, true)
  console.warn('[phone sign-in] the request did not finish', error)
  return ready(waiting, false, true)
}

function usePhoneSignIn(refusal: string | null) {
  const [shown, setShown] = useState<Shown>(refusal === null ? LOADING_STATE : said(refusal, true))

  useEffect(() => {
    if (refusal !== null) return undefined
    let live = true
    fetchPhoneWaiting()
      .then((waiting) => {
        if (!live) return
        setShown(waiting === null ? said(PHONE_NOTHING_WAITING) : ready(waiting))
      })
      .catch((error: unknown) => {
        console.warn('[phone sign-in] what waits could not be read', error)
        if (live) setShown(said(PHONE_FAILED, true))
      })
    return () => {
      live = false
    }
  }, [refusal])

  const act = async (waiting: PhoneWaiting, step: 'continue' | 'cancel') => {
    setShown(ready(waiting, true))
    try {
      if (step === 'cancel') {
        await cancelPhoneSignIn()
        setShown(said(PHONE_CANCELLED))
        return
      }
      const link = await continuePhoneSignIn()
      setShown(said(phoneOpening(waiting.app)))
      window.location.assign(link)
    } catch (error) {
      setShown(afterRefusal(error, waiting))
    }
  }
  return { shown, act }
}

type WaitingProps = {
  readonly shown: Extract<Shown, { kind: 'waiting' }>
  readonly act: ReturnType<typeof usePhoneSignIn>['act']
}

function Waiting({ shown, act }: WaitingProps) {
  const { waiting, busy, failed } = shown
  return (
    <>
      <p className="text-sm">{phoneSigningInAs(waiting)}</p>
      <p className="text-sm font-medium">{PHONE_ONLY_IF_YOU_STARTED}</p>
      {waiting.removed_before && <p className="text-sm text-muted-foreground">{PHONE_REMOVED_BEFORE}</p>}
      {failed && <p role="alert" className="text-sm text-destructive">{PHONE_FAILED}</p>}
      <div className="flex gap-3">
        <Button className="flex-1" disabled={busy} onClick={() => void act(waiting, 'continue')}>
          {PHONE_CONTINUE}
        </Button>
        <Button className="flex-1" variant="outline" disabled={busy} onClick={() => void act(waiting, 'cancel')}>
          {PHONE_CANCEL}
        </Button>
      </div>
    </>
  )
}

function PhoneCard() {
  const params = useSearchParams()
  const { shown, act } = usePhoneSignIn(phoneRefusalMessage(params.get(PHONE_ERROR_PARAM)))

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground">
      <section
        aria-labelledby="phone-sign-in-title"
        className="w-full max-w-sm space-y-5 rounded-2xl border border-border bg-card p-6 text-card-foreground"
      >
        <h1 id="phone-sign-in-title" className="text-xl font-semibold">{PHONE_TITLE}</h1>
        {shown.kind === 'loading' && <p className="text-sm text-muted-foreground">{LOADING}</p>}
        {shown.kind === 'said' && (
          <p role={shown.alert ? 'alert' : 'status'} className={shown.alert ? 'text-sm text-destructive' : 'text-sm'}>
            {shown.sentence}
          </p>
        )}
        {shown.kind === 'waiting' && <Waiting shown={shown} act={act} />}
      </section>
    </main>
  )
}

export function PhoneSignInConfirm() {
  return (
    <Suspense fallback={null}>
      <PhoneCard />
    </Suspense>
  )
}
