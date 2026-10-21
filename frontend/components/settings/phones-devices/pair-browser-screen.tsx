'use client'

/**
 * PRD-266 US-005b — "Pair this browser": a code field and a name, in place of the app, on a local
 * install whose anonymous lane is closed (LOCAL_REMOTE_ACCESS=paired). The code comes from
 * Settings → Phones & devices on a device already paired, or from the server console. A good code
 * stores this browser's device token (lib/device-token.ts) and the app shows; a refused one says
 * why in plain words, never the server's internals.
 */
import { useState, type FormEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { PAIRING_MESSAGES, pairingErrorMessage, pairThisBrowser, type PairedBrowser } from '@/hooks/use-mobile-devices'

const CONSOLE_COMMAND = 'docker compose exec backend python -m scripts.mint_pairing_code'
/** As api/mobile_devices.py takes them (PAIRING_CODE_MAX_LENGTH, NAME_LENGTH). */
const CODE_MAX_LENGTH = 32
const NAME_MAX_LENGTH = 120

interface PairBrowserScreenProps {
  /** Keeps the pairing; throws when the browser cannot (site storage blocked). */
  readonly onPaired: (answer: PairedBrowser) => void
}

/** Pair, then keep the pairing: the message to show when either step fails, or null. */
async function attemptPairing(code: string, name: string, onPaired: PairBrowserScreenProps['onPaired']): Promise<string | null> {
  let answer: PairedBrowser
  try {
    answer = await pairThisBrowser(code, name)
  } catch (failure) {
    return pairingErrorMessage(failure)
  }
  try {
    onPaired(answer)
    return null
  } catch (failure) {
    console.error('[pairing] this browser could not keep its pairing', failure)
    return failure instanceof Error ? failure.message : PAIRING_MESSAGES.failed
  }
}

export function PairBrowserScreen({ onPaired }: PairBrowserScreenProps) {
  const [code, setCode] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    const refused = await attemptPairing(code, name, onPaired)
    if (refused === null) return
    setError(refused)
    setBusy(false)
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground">
      <form
        onSubmit={submit}
        aria-labelledby="pair-browser-title"
        className="w-full max-w-sm space-y-5 rounded-2xl border border-border bg-card p-6 text-card-foreground"
      >
        <div className="space-y-2">
          <h1 id="pair-browser-title" className="text-xl font-semibold">Pair this browser</h1>
          <p className="text-sm text-muted-foreground">
            This install only answers paired devices. Enter a one-time code from Settings → Phones
            &amp; devices on a device already paired, or mint the first one on the server:
          </p>
          <code className="block break-all rounded-md bg-muted px-2 py-1 text-xs text-foreground">{CONSOLE_COMMAND}</code>
        </div>
        <div className="space-y-2">
          <Label htmlFor="pair-browser-code">Pairing code</Label>
          <Input
            id="pair-browser-code"
            value={code}
            onChange={(event) => setCode(event.target.value)}
            placeholder="XXXX-XXXX"
            autoComplete="one-time-code"
            autoCapitalize="characters"
            spellCheck={false}
            maxLength={CODE_MAX_LENGTH}
            required
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="pair-browser-name">Name this browser</Label>
          <Input
            id="pair-browser-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Office laptop"
            maxLength={NAME_MAX_LENGTH}
          />
        </div>
        {error !== null && <p role="alert" className="text-sm text-destructive">{error}</p>}
        <Button type="submit" className="w-full" disabled={busy || code.trim() === ''}>
          {busy ? 'Pairing…' : 'Pair'}
        </Button>
      </form>
    </main>
  )
}
