import Link from 'next/link'
import { ArrowLeft, Cloud } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

/**
 * PRD-233 S7 — what a SaaS-only route renders in the local edition.
 *
 * Route pages under `SAAS_ONLY_ROUTES` (lib/auth-edition) keep their hosted UI
 * byte-for-byte in `saas` and render this instead in `local`: the route still
 * resolves (a deep link or a stale bookmark lands somewhere honest), it just
 * does not pretend the surface exists. The copy is the owner's.
 */
export const SAAS_ONLY_NOTICE_COPY =
  'This area is part of the hosted edition; the local edition has no accounts, teams or plans.'

export function SaasOnlyNotice({ surface }: { surface: string }) {
  return (
    <div data-testid="saas-only-notice" className="flex items-center justify-center px-4 py-16">
      <Card className="w-full max-w-md">
        <CardContent className="space-y-4 p-8 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted">
            <Cloud className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
          </div>
          <h1 className="text-xl font-semibold text-foreground">{surface}</h1>
          <p className="text-sm text-muted-foreground">{SAAS_ONLY_NOTICE_COPY}</p>
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 text-sm font-medium text-primary hover:underline"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Back to chat
          </Link>
        </CardContent>
      </Card>
    </div>
  )
}
