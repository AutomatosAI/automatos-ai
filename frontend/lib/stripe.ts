/**
 * PRD-265 W2 — Stripe.js for the in-app payment forms (Settings → Billing).
 *
 * The card fields are Stripe's iframes inside our own form, so the page never leaves
 * Automatos. `getStripe` loads Stripe.js once per publishable key (the key comes from the
 * billing API, never from a build setting) and hands every caller the same promise: the
 * `stripe` prop of `<Elements>` / `<CheckoutElementsProvider>` must not change once set. It
 * never loads during a server render, and anything that is not a publishable key (an empty
 * value, a secret key) loads nothing.
 *
 * The client-secret checks keep a malformed API answer from reaching Stripe.js: a Checkout
 * Session secret starts `cs_`, a SetupIntent's `seti_`, a PaymentIntent's `pi_`, and each
 * carries `_secret_`.
 */
// The /pure entry adds Stripe's script only when loadStripe is first called (the Billing
// tab's card form), not on every page that imports this module.
import { loadStripe, type Stripe } from '@stripe/stripe-js/pure'

const PUBLISHABLE_KEY = /^pk_(?:test|live)_[A-Za-z0-9]+$/
const SECRET_MARKER = '_secret_'

export type ClientSecretKind = 'cs_' | 'seti_' | 'pi_'

const loaded = new Map<string, Promise<Stripe | null>>()

/** A Stripe publishable key (pk_test_… or pk_live_…), never a secret or restricted key. */
export function isPublishableKey(value: unknown): value is string {
  return typeof value === 'string' && PUBLISHABLE_KEY.test(value)
}

/** A client secret of the given kind ("cs_…_secret_…"). */
export function isClientSecret(value: unknown, kind: ClientSecretKind): value is string {
  return typeof value === 'string' && value.startsWith(kind) && value.includes(SECRET_MARKER)
}

/**
 * Stripe.js for `publishableKey`, loaded once per key. null on the server or for anything that
 * is not a publishable key. A failed load is forgotten, so the next attempt loads again.
 */
export function getStripe(publishableKey: string | null | undefined): Promise<Stripe | null> | null {
  if (typeof window === 'undefined' || !isPublishableKey(publishableKey)) return null
  const key: string = publishableKey
  const existing = loaded.get(key)
  if (existing) return existing
  const loading = loadStripe(key)
  loaded.set(key, loading)
  loading.catch(() => loaded.delete(key))
  return loading
}
