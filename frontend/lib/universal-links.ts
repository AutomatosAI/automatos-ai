/**
 * PRD-266 US-018c: the app-association files that let Automatos Go open this site's links.
 *
 * iOS fetches /.well-known/apple-app-site-association and Android /.well-known/assetlinks.json,
 * both signed out, so the saas middleware lists the two exact paths as public. Each is served in
 * both editions: a self-hosted web address is not one of the app's associated domains, so the
 * files are inert there. The app opens /go/*, Command Center and chat links; nothing else.
 */

export const IOS_APP_ID = 'KGVA9933BG.com.automatos.app'
export const ANDROID_PACKAGE = 'com.automatos.app'
export const APP_LINK_PATHS: readonly string[] = ['/go/*', '/command-center', '/command-center/*', '/chat', '/chat/*']

// The route reads the frontend server's ANDROID_APP_SHA256_CERT_FINGERPRINTS at request time: the
// EAS Android signing certificate's SHA-256, comma-separated (public values, not a secret).
// Unset, or nothing well-formed in it → assetlinks.json lists nothing.
const SHA256_CERT_FINGERPRINT = /^([0-9A-F]{2}:){31}[0-9A-F]{2}$/
const HANDLE_ALL_URLS = 'delegate_permission/common.handle_all_urls'

const SECONDS_PER_HOUR = 60 * 60
export const ASSOCIATION_MAX_AGE_S = SECONDS_PER_HOUR
export const ASSOCIATION_HEADERS: Readonly<Record<string, string>> = {
  'Content-Type': 'application/json',
  'Cache-Control': `public, max-age=${ASSOCIATION_MAX_AGE_S}`,
}

export interface AppleAppSiteAssociation {
  applinks: { details: { appIDs: string[]; components: { '/': string }[] }[] }
}

export interface AssetLinkStatement {
  relation: string[]
  target: { namespace: 'android_app'; package_name: string; sha256_cert_fingerprints: string[] }
}

export function appleAppSiteAssociation(): AppleAppSiteAssociation {
  const components = APP_LINK_PATHS.map((linkPath) => ({ '/': linkPath }))
  return { applinks: { details: [{ appIDs: [IOS_APP_ID], components }] } }
}

/** The well-formed fingerprints in a comma-separated value, in order and unique; the rest are dropped. */
export function androidCertFingerprints(raw: string | undefined): string[] {
  const valid = (raw ?? '')
    .split(',')
    .map((part) => part.trim())
    .filter((part) => SHA256_CERT_FINGERPRINT.test(part))
  return Array.from(new Set(valid))
}

/** The Digital Asset Links statements: one for the app, or none (a valid, empty list) without a fingerprint. */
export function assetLinks(fingerprints: readonly string[]): AssetLinkStatement[] {
  if (fingerprints.length === 0) return []
  return [
    {
      relation: [HANDLE_ALL_URLS],
      target: { namespace: 'android_app', package_name: ANDROID_PACKAGE, sha256_cert_fingerprints: [...fingerprints] },
    },
  ]
}

/** A 200 JSON answer with the association headers: never a redirect, cacheable by Apple's and Google's fetchers. */
export function associationResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { ...ASSOCIATION_HEADERS } })
}
