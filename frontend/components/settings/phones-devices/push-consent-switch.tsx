'use client'

/**
 * PRD-266 US-019a (D21, K13) — Settings → Phones & devices on a local install: the operator's
 * answer to "Send notifications through Expo's push service?" as a switch, so it can be changed
 * after "Pair a phone" asked it. It reads GET /api/v1/mobile/push-consent and writes PUT; the
 * answer is install-wide and discovery's push.expo follows it. phones-devices-tab.tsx shows it
 * only on a local install (SaaS always pushes through Expo); a caller the route refuses (403: not
 * an owner or admin) sees nothing.
 */
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { isConsentForbidden, useAnswerPushConsent, usePushConsent } from '@/hooks/use-push-consent'

import { ANSWER_FAILED, PUSH_CONSENT_EXPLAINED } from './push-consent-question'

export const PUSH_CONSENT_SWITCH_LABEL = 'Send notifications through Expo’s push service'
export const CONSENT_UNREADABLE = 'Couldn’t read this install’s answer right now.'
const SWITCH_ID = 'push-via-expo'

export function PushConsentSwitch() {
  const consent = usePushConsent()
  const answer = useAnswerPushConsent()
  if (isConsentForbidden(consent.error)) return null
  const readFailed = consent.isError
  return (
    <section aria-labelledby="push-consent-title" className="space-y-2 rounded-lg border border-border/40 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <Label id="push-consent-title" htmlFor={SWITCH_ID} className="text-sm font-medium text-foreground">
            {PUSH_CONSENT_SWITCH_LABEL}
          </Label>
          <p className="text-xs text-muted-foreground">{PUSH_CONSENT_EXPLAINED}</p>
        </div>
        <Switch
          id={SWITCH_ID}
          aria-labelledby="push-consent-title"
          checked={consent.data?.agreed ?? false}
          disabled={!consent.data || answer.isLoading}
          onCheckedChange={(agreed) => answer.mutate(agreed)}
        />
      </div>
      {readFailed && <p role="alert" className="text-sm text-destructive">{CONSENT_UNREADABLE}</p>}
      {answer.isError && <p role="alert" className="text-sm text-destructive">{ANSWER_FAILED}</p>}
    </section>
  )
}
