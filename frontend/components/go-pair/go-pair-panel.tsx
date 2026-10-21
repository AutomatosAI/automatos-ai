'use client'

/**
 * PRD-266 US-018d — the body of /go/pair: the code and the server from the link, how to pair from
 * inside Automatos Go, and how to get the app.
 *
 * A pairing QR scanned with the phone's camera always lands here (a self-hosted web address is never
 * one of the app's associated domains); the in-app scanner is what pairs. So this page never pairs:
 * it calls nothing, stores nothing, and shows the two query values as text only, each checked first
 * (pair-link.ts). A link that fails either check shows one plain message instead.
 */
import { useSearchParams } from 'next/navigation'
import { readPairLink, type PairLink } from '@/components/go-pair/pair-link'
import { GetAutomatosGo } from '@/components/go-pair/get-automatos-go'

export const INCOMPLETE_LINK_MESSAGE =
  'This link is incomplete. Ask for a new code in Studio (Settings → Phones & devices).'
// Anyone can make a link in this shape with any server in it, so the page never vouches for the
// server: it says what the link asks for, and the warning comes before the server's name.
export const TRUST_WARNING = 'Check that you know this server. Pair only with a server you run or trust.'

function PairLinkDetails({ link }: { link: PairLink }) {
  return (
    <section aria-labelledby="go-pair-details" className="space-y-4 rounded-lg border border-border bg-card p-5 text-card-foreground">
      <h2 id="go-pair-details" className="text-base font-semibold text-foreground">
        This link asks to pair the Automatos Go app with the server below
      </h2>
      <p data-testid="go-pair-trust" className="text-sm font-medium text-foreground">
        {TRUST_WARNING}
      </p>
      <dl className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <dt className="text-xs uppercase tracking-wide text-muted-foreground">Server</dt>
          <dd data-testid="go-pair-server" className="break-all font-mono text-foreground">
            {link.serverHost}
          </dd>
        </div>
        <div className="space-y-1">
          <dt className="text-xs uppercase tracking-wide text-muted-foreground">Code</dt>
          <dd data-testid="go-pair-code" className="font-mono text-lg tracking-widest text-foreground">
            {link.code}
          </dd>
        </div>
      </dl>
    </section>
  )
}

function HowToPair() {
  return (
    <section aria-labelledby="go-pair-how" className="space-y-2">
      <h2 id="go-pair-how" className="text-base font-semibold text-foreground">
        Pair from inside the app
      </h2>
      <ol className="list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
        <li>Open Automatos Go on this phone and scan the QR code again, from inside the app.</li>
        <li>Or choose ‘Connect to your own server’ and type the address and the code above.</li>
      </ol>
      <p className="text-sm text-muted-foreground">
        This browser does not pair by itself. If the code has expired, ask for a new one in Studio
        (Settings → Phones &amp; devices).
      </p>
    </section>
  )
}

/** The link's details and the steps when the link checks out, else the incomplete-link message. */
export function GoPairPanel() {
  const link = readPairLink(useSearchParams())
  return (
    <div className="space-y-8">
      {link ? (
        <>
          <PairLinkDetails link={link} />
          <HowToPair />
        </>
      ) : (
        <p data-testid="go-pair-incomplete" className="rounded-lg border border-border bg-card p-5 text-sm text-foreground">
          {INCOMPLETE_LINK_MESSAGE}
        </p>
      )}
      <GetAutomatosGo />
    </div>
  )
}
