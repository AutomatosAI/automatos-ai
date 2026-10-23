'use client'

/**
 * PRD-266 US-005c — "Pair a phone" and "Pair a browser" in Settings → Phones & devices, shown only
 * on a local install in paired mode (LOCAL_REMOTE_ACCESS=paired; phones-devices-tab.tsx decides).
 *
 * Pair a phone mints a one-time code (POST /api/v1/mobile/pairing-codes) and shows the code, its
 * ten-minute countdown and its pairing link, the universal link
 * `<web address>/go/pair?api=<API address>&code=<code>`, as text with a copy button. The QR image
 * of that link is US-019 (K9: the repo has no QR encoder). The same code pairs another browser:
 * Pair a browser says how.
 *
 * A browser that reaches the install over plain http on the LAN is not a secure context and has
 * no navigator.clipboard, so Copy can fail there: it then says to select the link, which stays on
 * screen as text.
 */
import { useState } from 'react'
import { Check, Copy, Loader2, Monitor, Smartphone } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { PairingCode } from '@/hooks/use-mobile-devices'
import { copyToClipboard } from '@/lib/utils'

const SECONDS_PER_MINUTE = 60
const COPIED_FOR_MS = 1500
export const CODE_EXPIRED = 'This code has expired. Get a new one.'
export const COPY_REFUSED = 'This browser won’t copy from here: select the link and copy it.'

/** Seconds as m:ss, the way a countdown reads (600 → "10:00", 65 → "1:05"). */
export function formatCountdown(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  const rest = String(whole % SECONDS_PER_MINUTE).padStart(2, '0')
  return `${Math.floor(whole / SECONDS_PER_MINUTE)}:${rest}`
}

function PairingLink({ link }: { link: string }) {
  const [copy, setCopy] = useState<'idle' | 'copied' | 'refused'>('idle')
  const copyLink = async () => {
    const copied = await copyToClipboard(link)
    setCopy(copied ? 'copied' : 'refused')
    if (copied) setTimeout(() => setCopy('idle'), COPIED_FOR_MS)
  }
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">
        Pairing link (it carries this install&apos;s API address and the code):
      </p>
      <div className="flex items-center gap-2">
        <code className="flex-1 select-all break-all rounded bg-muted px-3 py-2 font-mono text-xs text-foreground" data-testid="pairing-link">
          {link}
        </code>
        <Button type="button" variant="outline" size="sm" onClick={copyLink} aria-label="Copy the pairing link">
          {copy === 'copied' ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        </Button>
      </div>
      {copy === 'refused' && <p role="status" className="text-xs text-muted-foreground">{COPY_REFUSED}</p>}
    </div>
  )
}

function LiveCode({ pairing, secondsLeft }: { pairing: PairingCode; secondsLeft: number }) {
  if (secondsLeft <= 0) return <p className="text-sm text-muted-foreground">{CODE_EXPIRED}</p>
  return (
    <div className="space-y-3 text-sm">
      <p className="text-muted-foreground">Enter this code in Automatos Go on the phone. It works once.</p>
      <p className="font-mono text-2xl font-semibold tracking-widest text-foreground" data-testid="pairing-code">
        {pairing.code}
      </p>
      <p className="text-xs text-muted-foreground" data-testid="pairing-countdown">
        Expires in {formatCountdown(secondsLeft)}
      </p>
      <PairingLink link={pairing.qr_payload} />
    </div>
  )
}

interface PairPhonePanelProps {
  readonly pairing: PairingCode | undefined
  readonly secondsLeft: number | null
  readonly minting: boolean
  readonly error: string | null
  readonly onMint: () => void
}

export function PairPhonePanel({ pairing, secondsLeft, minting, error, onMint }: PairPhonePanelProps) {
  const label = pairing ? 'Get a new code' : 'Get a pairing code'
  return (
    <section aria-labelledby="pair-phone-title" className="space-y-3 rounded-lg border border-border/40 p-4">
      <div className="flex items-center gap-2">
        <Smartphone className="h-4 w-4 text-muted-foreground" aria-hidden />
        <h3 id="pair-phone-title" className="text-sm font-medium text-foreground">Pair a phone</h3>
      </div>
      <p className="text-xs text-muted-foreground">
        A one-time code pairs Automatos Go on a phone to you, in this workspace. It works once, for
        ten minutes.
      </p>
      <Button type="button" onClick={onMint} disabled={minting}>
        {minting ? <Loader2 className="h-4 w-4 animate-spin" /> : label}
      </Button>
      {error !== null && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {pairing && secondsLeft !== null && <LiveCode pairing={pairing} secondsLeft={secondsLeft} />}
    </section>
  )
}

export function PairBrowserNote({ webUrl }: { webUrl: string | null }) {
  return (
    <section aria-labelledby="pair-browser-note-title" className="space-y-2 rounded-lg border border-border/40 p-4">
      <div className="flex items-center gap-2">
        <Monitor className="h-4 w-4 text-muted-foreground" aria-hidden />
        <h3 id="pair-browser-note-title" className="text-sm font-medium text-foreground">Pair a browser</h3>
      </div>
      <p className="text-xs text-muted-foreground">
        In paired mode every browser pairs too. Open{' '}
        {webUrl ? <span className="font-mono text-foreground">{webUrl}</span> : 'this web app'} in the
        other browser: it asks for a code. Get one above and enter it there, with a name for that
        browser.
      </p>
    </section>
  )
}
