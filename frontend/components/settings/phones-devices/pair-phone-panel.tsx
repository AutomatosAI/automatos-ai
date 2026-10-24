'use client'

/**
 * PRD-266 US-005c — "Pair a phone" and "Pair a browser" in Settings → Phones & devices, shown only
 * on a local install in paired mode (LOCAL_REMOTE_ACCESS=paired; phones-devices-tab.tsx decides).
 *
 * Pair a phone mints a one-time code (POST /api/v1/mobile/pairing-codes) and shows the code, its
 * ten-minute countdown and its pairing link, the universal link
 * `<web address>/go/pair?api=<API address>&code=<code>`, as text with a copy button. US-019c (K17):
 * the QR image of that link sits beside them (pair-phone-qr.tsx). The same code pairs another
 * browser: Pair a browser says how.
 *
 * US-019a: the code is shown exactly as the server returns it, XXXX-XXXX with its hyphen, and its
 * copy button copies that form (POST /pair canonicalises a code typed with or without it). Before
 * the first code on an install whose operator never answered the Expo question, the panel shows
 * that question (its children, from phones-devices-tab.tsx).
 *
 * A browser that reaches the install over plain http on the LAN is not a secure context and has
 * no navigator.clipboard, so Copy can fail there: it then says to select the link, which stays on
 * screen as text.
 */
import { useState, type ReactNode } from 'react'
import { Check, Copy, Loader2, Monitor, Smartphone } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { PairingCode } from '@/hooks/use-mobile-devices'
import { copyToClipboard } from '@/lib/utils'

import { PairPhoneQr } from './pair-phone-qr'

const SECONDS_PER_MINUTE = 60
const COPIED_FOR_MS = 1500
export const CODE_EXPIRED = 'This code has expired. Get a new one.'
export const COPY_REFUSED = 'This browser won’t copy from here: select the link and copy it.'
export const CODE_COPY_REFUSED = 'This browser won’t copy from here: select the code and copy it.'

type CopyState = 'idle' | 'copied' | 'refused'

/** Seconds as m:ss, the way a countdown reads (600 → "10:00", 65 → "1:05"). */
export function formatCountdown(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  const rest = String(whole % SECONDS_PER_MINUTE).padStart(2, '0')
  return `${Math.floor(whole / SECONDS_PER_MINUTE)}:${rest}`
}

/** Copy `text`; 'refused' where the browser has no clipboard (plain http on the LAN). */
function useCopy(text: string): { readonly copy: CopyState; readonly onCopy: () => Promise<void> } {
  const [copy, setCopy] = useState<CopyState>('idle')
  const onCopy = async () => {
    const copied = await copyToClipboard(text)
    setCopy(copied ? 'copied' : 'refused')
    if (copied) setTimeout(() => setCopy('idle'), COPIED_FOR_MS)
  }
  return { copy, onCopy }
}

function CopyButton({ copy, onCopy, label }: { copy: CopyState; onCopy: () => void; label: string }) {
  return (
    <Button type="button" variant="outline" size="sm" onClick={onCopy} aria-label={label}>
      {copy === 'copied' ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </Button>
  )
}

function PairingCodeLine({ code }: { code: string }) {
  const { copy, onCopy } = useCopy(code)
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-3">
        <p className="font-mono text-2xl font-semibold tracking-widest text-foreground" data-testid="pairing-code">
          {code}
        </p>
        <CopyButton copy={copy} onCopy={onCopy} label="Copy the code" />
      </div>
      {copy === 'refused' && <p role="status" className="text-xs text-muted-foreground">{CODE_COPY_REFUSED}</p>}
    </div>
  )
}

function PairingLink({ link }: { link: string }) {
  const { copy, onCopy } = useCopy(link)
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">
        Pairing link (it carries this install&apos;s API address and the code):
      </p>
      <div className="flex items-center gap-2">
        <code className="flex-1 select-all break-all rounded bg-muted px-3 py-2 font-mono text-xs text-foreground" data-testid="pairing-link">
          {link}
        </code>
        <CopyButton copy={copy} onCopy={onCopy} label="Copy the pairing link" />
      </div>
      {copy === 'refused' && <p role="status" className="text-xs text-muted-foreground">{COPY_REFUSED}</p>}
    </div>
  )
}

function LiveCode({ pairing, secondsLeft }: { pairing: PairingCode; secondsLeft: number }) {
  if (secondsLeft <= 0) return <p className="text-sm text-muted-foreground">{CODE_EXPIRED}</p>
  return (
    <div className="flex flex-wrap items-start gap-4 text-sm">
      <PairPhoneQr payload={pairing.qr_payload} />
      <div className="min-w-0 flex-1 basis-64 space-y-3">
        <p className="text-muted-foreground">
          Scan the QR code with the phone, or enter this code in Automatos Go on it. It works once.
        </p>
        <PairingCodeLine code={pairing.code} />
        <p className="text-xs text-muted-foreground" data-testid="pairing-countdown">
          Expires in {formatCountdown(secondsLeft)}
        </p>
        <PairingLink link={pairing.qr_payload} />
      </div>
    </div>
  )
}

interface PairPhonePanelProps {
  readonly pairing: PairingCode | undefined
  readonly secondsLeft: number | null
  readonly minting: boolean
  readonly error: string | null
  readonly onMint: () => void
  /** Shown under the button: the Expo question, before the first code (US-019a). */
  readonly children?: ReactNode
}

export function PairPhonePanel({ pairing, secondsLeft, minting, error, onMint, children }: PairPhonePanelProps) {
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
      {children}
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
