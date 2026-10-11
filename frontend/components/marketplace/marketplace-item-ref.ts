/** Agents and playbooks have separate id sequences: an item is its type plus its id. */
export interface MarketplaceItemRef {
  id: number
  /** 'agent' or 'recipe' (a playbook), as /api/marketplace/items returns it. */
  type: string
}

export const marketplaceItemKey = (item: MarketplaceItemRef) => `${item.type}:${item.id}`
