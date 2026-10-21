'use client'

/**
 * PRD-266 US-018d — /go/pair, where a pairing QR opens when it is scanned with the phone's camera.
 *
 * Public in both editions: the saas middleware lists the exact path /go/pair, and the local edition
 * serves every route. No sign-in, no app shell, no workspace data, and no API call of any kind: it
 * repeats the code and the server from the link and says how to pair from inside Automatos Go.
 */
import { PageHeader } from '@/components/shared/page-header'
import { GoPairPanel } from '@/components/go-pair/go-pair-panel'

export default function GoPairPage() {
  return (
    <main className="min-h-screen bg-background px-4 py-10 text-foreground">
      <div className="mx-auto max-w-2xl space-y-8">
        <PageHeader
          eyebrow="Automatos Go"
          title="Pair your"
          titleAccent="phone"
          lede="This link pairs the Automatos Go app with an Automatos server. The app does the pairing, so open it on this phone to finish."
        />
        <GoPairPanel />
      </div>
    </main>
  )
}
