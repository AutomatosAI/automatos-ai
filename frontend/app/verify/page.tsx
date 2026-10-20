'use client'

/**
 * PRD-264 EUAI-10 — /verify, the public check for an Automatos AI mark (Article 50(2)).
 *
 * Public in both editions: the saas middleware lists /verify(.*) as a public route and
 * the local edition serves every route. It is a host-owned route (HOST_OWNED_ROUTES), so
 * no theme can replace it. No sign-in, no app shell, no workspace data.
 */
import { PageHeader } from '@/components/shared/page-header'
import { ProvenanceCheck } from '@/components/verify/provenance-check'

export default function VerifyPage() {
  return (
    <main className="min-h-screen bg-background px-4 py-10 text-foreground">
      <div className="mx-auto max-w-2xl space-y-8">
        <PageHeader
          eyebrow="Automatos AI"
          title="Check a"
          titleAccent="file"
          lede="See whether a video, image, audio file or document was made by Automatos AI. The check uses the file's fingerprint, so the file itself is never uploaded."
        />
        <ProvenanceCheck />
      </div>
    </main>
  )
}
