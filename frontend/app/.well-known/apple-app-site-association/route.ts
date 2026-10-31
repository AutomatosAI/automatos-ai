import { appleAppSiteAssociation, associationResponse } from '@/lib/universal-links'

// PRD-266 US-018c: iOS reads this, signed out, to open Automatos Go for /go/*, Command Center and chat links.
// Answered per request, as assetlinks.json is, so the response carries the handler's own Cache-Control.
export const dynamic = 'force-dynamic'

export function GET(): Response {
  return associationResponse(appleAppSiteAssociation())
}
