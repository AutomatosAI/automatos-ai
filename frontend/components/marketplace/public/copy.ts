/**
 * PRD-270 (§5.7, one Marketplace page for everyone) — the words a signed-out visitor reads on the
 * Marketplace: the header line under "Community Marketplace", Popular in place of the personal
 * "Recommended for You", and the public detail and the visitor's tabs. Plain words, no hype. The
 * Themes storefront keeps its own (../themes/listing-format.ts, ../themes/storefront/copy.ts).
 */

export const VISITOR_EYEBROW = 'Automatos AI · Marketplace'
export const VISITOR_LEDE =
  'Themes, agents, Playbooks, packages and tools for Automatos. Browse all of it here. To add any of it to a workspace, sign in, or choose Start free to make one.'
/** Signed out, the strip after Featured: the most installed items besides the featured ones. */
export const POPULAR_TITLE = 'Popular'

export const LOADING_ITEMS = 'Loading…'
export const NO_ITEMS = 'Nothing is listed here yet.'
export const NO_ITEM_MATCHES = 'Nothing here matches your search.'

export const REQUIRED_APPS = 'Needs these apps connected'
export const IN_THIS_PACKAGE = 'In this package'
export const CAPABILITIES = 'Capabilities'
export const TAGS = 'Tags'

export function byPublisher(name: string): string {
  return `By ${name}`
}
