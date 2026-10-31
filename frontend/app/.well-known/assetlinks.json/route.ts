import { androidCertFingerprints, assetLinks, associationResponse } from '@/lib/universal-links'

// PRD-266 US-018c: Android reads this, signed out, to open Automatos Go for the same links as iOS.
// The fingerprints are the server's env at request time, so the build never freezes them.
export const dynamic = 'force-dynamic'

export function GET(): Response {
  const fingerprints = androidCertFingerprints(process.env.ANDROID_APP_SHA256_CERT_FINGERPRINTS)
  return associationResponse(assetLinks(fingerprints))
}
