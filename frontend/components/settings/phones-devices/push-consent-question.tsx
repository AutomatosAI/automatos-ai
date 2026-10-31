'use client'

/**
 * PRD-266 US-019a (D21, K13) — the question "Pair a phone" asks once on a local install, before
 * its first code: may this install send notifications to phones through Expo's push service?
 * Yes or Not now; either answer is saved (PUT /api/v1/mobile/push-consent), then the code is
 * minted. hooks/use-push-consent.ts decides when it shows; the answer can be changed later with
 * the switch on Settings → Phones & devices (push-consent-switch.tsx).
 */
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'

export const PUSH_CONSENT_QUESTION = 'Send notifications to phones through Expo’s push service?'
/** What saying yes means: one line, shared with the switch. */
export const PUSH_CONSENT_EXPLAINED =
  'Expo relays each notification to the phone. A question’s text travels in it unless this workspace redacts questions.'
export const ANSWER_FAILED = 'Couldn’t save your answer. Try again.'

interface PushConsentQuestionProps {
  readonly saving: boolean
  readonly failed: boolean
  readonly onAnswer: (agreed: boolean) => void
}

export function PushConsentQuestion({ saving, failed, onAnswer }: PushConsentQuestionProps) {
  return (
    <div role="group" aria-labelledby="push-consent-question" className="space-y-2 rounded-md border border-border/40 bg-muted/40 p-3">
      <p id="push-consent-question" className="text-sm font-medium text-foreground">{PUSH_CONSENT_QUESTION}</p>
      <p className="text-xs text-muted-foreground">{PUSH_CONSENT_EXPLAINED}</p>
      <div className="flex items-center gap-2">
        <Button type="button" size="sm" onClick={() => onAnswer(true)} disabled={saving}>Yes</Button>
        <Button type="button" size="sm" variant="outline" onClick={() => onAnswer(false)} disabled={saving}>
          Not now
        </Button>
        {saving && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-hidden />}
      </div>
      {failed && <p role="alert" className="text-sm text-destructive">{ANSWER_FAILED}</p>}
    </div>
  )
}
