/**
 * 2026-10-09: how a model's price reads in the pickers. A route the catalogue has no
 * price for (the Anthropic sync lists a new model without one) used to read "free";
 * it costs money on the owner's key, so it reads "price unknown".
 */

export const PRICE_UNKNOWN = 'Price unknown'

export interface ModelPriceFacts {
  is_free?: boolean
  /** False: the catalogue has no price for this route. */
  price_known?: boolean
}

/** The suffix after a model's name in a dropdown: " · free", " · price unknown" or nothing. */
export function priceNote(model: ModelPriceFacts): string {
  if (model.is_free) return ' · free'
  return model.price_known === false ? ` · ${PRICE_UNKNOWN.toLowerCase()}` : ''
}
