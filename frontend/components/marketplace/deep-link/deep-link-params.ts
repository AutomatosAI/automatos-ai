/**
 * PRD-270 US-270-09b (§5.7, §8-Q8, K22) — the Marketplace link that opens one item,
 * `/marketplace?tab=<tab>&item=<key>&add=1` (lib/marketplace/return-after-sign-in.ts
 * `marketplaceItemPath`), as a member lands on it after sign-in or sign-up, or follows it.
 *
 * `tab` names the item's tab as the tab strip reads it (`openedMarketplaceTab`: a value or a label,
 * `plugins` or `skills` for a Capabilities sub-tab, none for Themes); `item` is a theme's listing id
 * or the public read's key; `add=1` asks for the add dialog. Reading a link installs nothing: the
 * person still clicks Add.
 */
import {
  ADD_PARAM,
  ADD_READY,
  ITEM_PARAM,
  MARKETPLACE_PATH,
  MAX_PARAM_LENGTH,
  TAB_PARAM,
} from '@/lib/marketplace/return-after-sign-in'

import { CAPABILITIES_TAB_VALUE, openedMarketplaceTab } from '../category-tabs'

/** What a link is read from: the page's search params (`useSearchParams`, null outside the app router). */
export type LinkParams = Pick<URLSearchParams, 'get' | 'toString'>

export interface MarketplaceLink {
  /** The item's tab: a CATEGORY_TABS value, or `plugins` / `skills` for a Capabilities sub-tab. */
  readonly tab: string
  readonly item: string
  /** `add=1`: open the item's add dialog. */
  readonly add: boolean
}

/** The link the page's address carries, or null when it names no item (or an overlong one). */
export function marketplaceLink(params: LinkParams | null | undefined): MarketplaceLink | null {
  const item = params?.get(ITEM_PARAM)?.trim()
  if (!params || !item || item.length > MAX_PARAM_LENGTH) return null
  const opened = openedMarketplaceTab(params.get(TAB_PARAM))
  const tab = opened.tab === CAPABILITIES_TAB_VALUE ? opened.capability : opened.tab
  return { tab, item, add: params.get(ADD_PARAM) === ADD_READY }
}

/**
 * The Marketplace's address without the link's `item` and `add`, every other parameter kept (its
 * tab among them): where closing the dialog goes, so a reload doesn't reopen it.
 */
export function pathWithoutLink(params: LinkParams | null | undefined): string {
  const kept = Array.from(new URLSearchParams(params?.toString() ?? '')).filter(
    ([name]) => name !== ITEM_PARAM && name !== ADD_PARAM,
  )
  const query = new URLSearchParams(kept).toString()
  return query ? `${MARKETPLACE_PATH}?${query}` : MARKETPLACE_PATH
}
