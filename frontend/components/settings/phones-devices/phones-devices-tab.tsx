'use client'

/**
 * PRD-266 US-005c — Settings → Phones & devices, in both editions (§4.3 "Web"): the device list
 * with Revoke and, on a local install in paired mode (the discovery document says so), Pair a
 * phone and Pair a browser. In SaaS, phones sign in with their account, so there is no pairing
 * and the list shows the member's own phones (an owner or admin sees the workspace's). Every call
 * goes through hooks/use-mobile-devices.ts. While a code is live the list is read again every few
 * seconds, so the phone or browser that pairs with it appears.
 *
 * US-019a: a local install also shows the operator's Expo answer as a switch, and Pair a phone
 * asks it once, before the first code, while it was never answered (hooks/use-push-consent.ts).
 *
 * PRD-277 US-601: an enterprise install shows Pair a phone while discovery says it lets people
 * pair, to the signed-in person, for themselves (the code pairs to its maker), with no Pair a
 * browser (an enterprise browser signs in). The Expo switch shows to an install admin only: the
 * route refuses everyone else.
 */
import { useCountdown } from '@/hooks/use-countdown'
import { useMintPairingCode, useMobileDevices, usePairingMode } from '@/hooks/use-mobile-devices'
import { useAskOnceBeforePairing } from '@/hooks/use-push-consent'
import { isLocal } from '@/lib/auth-edition'
import { isEnterprise } from '@/lib/edition-name'

import { DeviceList } from './device-list'
import { PairBrowserNote, PairPhonePanel } from './pair-phone-panel'
import { PushConsentQuestion } from './push-consent-question'
import { PushConsentSwitch } from './push-consent-switch'

/** How often the list is read while a pairing code is live. */
export const LIVE_CODE_REFRESH_MS = 5000
export const MINT_FAILED = 'Couldn’t get a pairing code'
/** Only a local install pairs browsers; a SaaS browser signs in. */
const TAB_LEDE = `The phones that use Automatos Go with this workspace${isLocal ? ', and the browsers paired to it' : ''}. ` +
  'You see your own; a workspace owner or admin sees everyone’s.'

/** The mint's failure as the panel says it: the server's own words when it gave some. */
function mintError(error: unknown): string | null {
  if (!error) return null
  return error instanceof Error && error.message ? `${MINT_FAILED}: ${error.message}` : `${MINT_FAILED}.`
}

export function PhonesDevicesTab() {
  const { paired, webUrl } = usePairingMode()
  const mint = useMintPairingCode()
  const question = useAskOnceBeforePairing(() => mint.mutate())
  const secondsLeft = useCountdown(mint.data?.expires_at)
  const codeLive = secondsLeft !== null && secondsLeft > 0
  const devices = useMobileDevices(codeLive ? LIVE_CODE_REFRESH_MS : undefined)

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-foreground">Phones &amp; devices</h2>
        <p className="text-sm text-muted-foreground">{TAB_LEDE}</p>
      </div>
      <DeviceList devices={devices.data} loading={devices.isLoading} failed={devices.isError} />
      {(isLocal || isEnterprise) && <PushConsentSwitch />}
      {paired && (
        <PairPhonePanel
          pairing={mint.data}
          secondsLeft={secondsLeft}
          minting={mint.isLoading || question.waiting || question.saving}
          error={mintError(mint.error)}
          onMint={question.start}
        >
          {question.asking && (
            <PushConsentQuestion saving={question.saving} failed={question.failed} onAnswer={question.reply} />
          )}
        </PairPhonePanel>
      )}
      {paired && isLocal && <PairBrowserNote webUrl={webUrl} />}
    </div>
  )
}
