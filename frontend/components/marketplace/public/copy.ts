/**
 * PRD-270 US-270-08b (§5.7) — the words of the signed-out visitor's Marketplace tabs. The Themes
 * storefront keeps its own (../themes/listing-format.ts, ../themes/storefront/copy.ts).
 */

export const VISITOR_LEDE =
  'Themes, agents, Playbooks, packages and tools for Automatos. Browse everything here; sign in to add any of it to your workspace.'
export const VISITOR_SEARCH = 'Search the Marketplace…'
export const VISITOR_TABS_LABEL = 'Marketplace tabs'
export const CAPABILITY_TABS_LABEL = 'Capabilities'

export const LOADING_ITEMS = 'Loading…'
export const NO_ITEMS = 'Nothing is listed here yet.'
export const NO_ITEM_MATCHES = 'Nothing here matches your search.'
export const SHOW_MORE = 'Show more'
export const LOADING_MORE = 'Loading more…'

export const REQUIRED_APPS = 'Needs these apps connected'
export const IN_THIS_PACKAGE = 'In this package'
export const CAPABILITIES = 'Capabilities'
export const TAGS = 'Tags'

export function byPublisher(name: string): string {
  return `By ${name}`
}

export function itemDetails(name: string): string {
  return `${name}: details`
}
