/**
 * PRD-222 W2·S1b (US-024) — marketplace plan-label chips.
 *
 * The full catalog stays VISIBLE to every tier (hidden ≠ deleted, D5). Items
 * that need a higher tier than the workspace's `marketplace_depth` get a small
 * plan-label chip ('Pro' / 'Business') — a label only, no install blocking.
 *
 * An item's required tier is DATA-DRIVEN from its own metadata, so re-tiering an
 * item (or seeding tiers) is a data change, never a code change.
 */

/** depth → label. 1 (curated/basic) needs no chip; 2 = Pro, 3 = Business. */
const DEPTH_LABEL: Record<number, string> = { 2: 'Pro', 3: 'Business' }

interface ItemWithMetadata {
  metadata?: Record<string, unknown> | null
}

/**
 * The depth level a marketplace item requires (1 = available to all tiers).
 * Read from the item's metadata: an explicit numeric `marketplace_depth` /
 * `min_depth` wins; otherwise a named `min_plan` / `tier` ('pro'|'business')
 * maps to a level. Anything else ⇒ 1.
 */
export function itemRequiredDepth(item: ItemWithMetadata): number {
  const m = item?.metadata || {}
  const raw = (m['marketplace_depth'] ?? m['min_depth']) as unknown
  if (typeof raw === 'number' && raw >= 1) return raw
  const plan = String(m['min_plan'] ?? m['tier'] ?? '').toLowerCase()
  if (plan === 'business') return 3
  if (plan === 'pro') return 2
  return 1
}

/**
 * The plan-label chip to show when an item is beyond the workspace's depth, or
 * null when the item is within reach (no chip, and never hidden — D5).
 */
export function planChipForItem(
  item: ItemWithMetadata,
  workspaceDepth: number,
): string | null {
  const need = itemRequiredDepth(item)
  if (need <= (workspaceDepth || 1)) return null
  return DEPTH_LABEL[need] ?? null
}
