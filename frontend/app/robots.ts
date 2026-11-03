/**
 * PRD-270 US-270-10 (§5.7 Search engines, K23) — /robots.txt.
 *
 * SaaS: the public Marketplace is indexable and nothing else is. `Allow: /marketplace` covers the
 * page and every tab and item link on it (`/marketplace?tab=…&item=…`); the longer
 * `Disallow: /marketplace/` keeps its signed-in pages (publish, developer, widgets) out, and
 * `Disallow: /` everything else. Two things the Marketplace needs are allowed with it: the logo a
 * share card shows (share-card.ts LOGO_PATH), as the crawlers that draw a card read robots.txt
 * before they fetch its picture, and the app's built scripts and styles (`/_next/static/`), as the
 * visitor's storefront draws in the browser and a search engine renders it with them.
 *
 * Local: an operator's own install, so nothing is indexed.
 *
 * The middleware's matcher skips /robots.txt (a path with an extension), so it is served signed out.
 */
import type { MetadataRoute } from 'next'

import { isSaaS } from '@/lib/auth-edition'
import { LOGO_PATH } from '@/lib/marketplace/share-card'
import { MARKETPLACE_PATH } from '@/lib/marketplace/return-after-sign-in'

const EVERY_CRAWLER = '*'
const EVERYTHING = '/'
/** Next's built scripts, styles and self-hosted fonts. */
const BUILT_ASSETS = '/_next/static/'

export default function robots(): MetadataRoute.Robots {
  if (!isSaaS) return { rules: { userAgent: EVERY_CRAWLER, disallow: EVERYTHING } }
  return {
    rules: {
      userAgent: EVERY_CRAWLER,
      allow: [MARKETPLACE_PATH, LOGO_PATH, BUILT_ASSETS],
      disallow: [EVERYTHING, `${MARKETPLACE_PATH}/`],
    },
  }
}
