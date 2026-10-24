'use client'

/**
 * PRD-266 US-019a (§4.3 "Web", §4.4, D21, K13) — the operator's answer to "Send notifications
 * through Expo's push service?", called only through apiClient.
 *
 * GET /api/v1/mobile/push-consent reads the answer ({agreed, answered, answered_at}) and PUT
 * records it ({agreed}). Both routes are local edition only (404 on SaaS, where push always goes
 * through the platform's Expo project), so SaaS never reads them; both are 403 for anyone but an
 * owner or admin. The answer is install-wide, and discovery's push.expo reads the same stored
 * setting (services/push/consent.py), so it says the same after a change.
 *
 * Settings → Phones & devices shows the answer as a switch (push-consent-switch.tsx), and "Pair a
 * phone" asks once, before its first code, while the answer was never given
 * (useAskOnceBeforePairing). Both read one cached answer: once the question is answered, the
 * switch shows it and the question is never asked again. A read that fails (403, 503) asks
 * nothing: the server never shows "could not read" as "not answered", and neither does Studio.
 */
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { isLocal } from '@/lib/auth-edition'

export const PUSH_CONSENT_PATH = '/api/v1/mobile/push-consent'
export const PUSH_CONSENT_QUERY_KEY = ['mobile-push-consent'] as const
const FORBIDDEN = 403
/** A failed read is tried once more, as the app's QueryClient does (components/providers.tsx). */
const READ_RETRIES = 1

/** The operator's answer as GET and PUT /push-consent return it. */
export interface PushConsent {
  readonly agreed: boolean
  readonly answered: boolean
  readonly answered_at: string | null
}

/** What "Pair a phone" needs to ask once before its first code. */
export interface PairingQuestion {
  /** The question is on screen: the code is minted once it is answered. */
  readonly asking: boolean
  /** The answer is still being read: minting waits for it. */
  readonly waiting: boolean
  /** The answer is being saved. */
  readonly saving: boolean
  /** The answer could not be saved: the question stays, to answer again. */
  readonly failed: boolean
  /** "Get a pairing code": ask first while the answer was never given, else mint. */
  readonly start: () => void
  /** Yes (true) or Not now (false): PUT the answer, then mint. */
  readonly reply: (agreed: boolean) => void
}

/** Did the route refuse this caller (403: not an owner or admin)? */
export function isConsentForbidden(error: unknown): boolean {
  return (error as ApiHttpError | null)?.status === FORBIDDEN
}

/** The operator's answer; read on a local install only. A 403 is final: it is not tried again. */
export function usePushConsent() {
  return useQuery({
    queryKey: PUSH_CONSENT_QUERY_KEY,
    queryFn: () => apiClient.get<PushConsent>(PUSH_CONSENT_PATH),
    enabled: isLocal,
    retry: (failures, error) => !isConsentForbidden(error) && failures < READ_RETRIES,
  })
}

/** Record the answer; the cached answer becomes the one the server stored. */
export function useAnswerPushConsent() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (agreed: boolean) => apiClient.put<PushConsent>(PUSH_CONSENT_PATH, { agreed }),
    onSuccess: (answer) => queryClient.setQueryData(PUSH_CONSENT_QUERY_KEY, answer),
  })
}

/** "Pair a phone" on a local install: ask the operator once, then mint with `mint`. */
export function useAskOnceBeforePairing(mint: () => void): PairingQuestion {
  const consent = usePushConsent()
  const answer = useAnswerPushConsent()
  const [asking, setAsking] = useState(false)
  const notAnswered = consent.data?.answered === false
  const start = () => (notAnswered ? setAsking(true) : mint())
  const reply = (agreed: boolean) => {
    answer.mutate(agreed, {
      onSuccess: () => {
        setAsking(false)
        mint()
      },
    })
  }
  return {
    // An answer given meanwhile (the switch, another tab) ends the question.
    asking: asking && notAnswered,
    waiting: consent.isInitialLoading,
    saving: answer.isLoading,
    failed: answer.isError,
    start,
    reply,
  }
}
