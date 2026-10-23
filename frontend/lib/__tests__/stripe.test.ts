/**
 * PRD-265 W2 — loading Stripe.js for the in-app payment forms: once per publishable key (the
 * `stripe` prop of Stripe's providers must not change once set), never during a server render,
 * never for anything that is not a publishable key, and again after a failed load. Client
 * secrets are checked by kind before they reach Stripe.js. The keys here are placeholders.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const { loadStripeMock } = vi.hoisted(() => ({ loadStripeMock: vi.fn(async (key: string) => ({ key })) }))

vi.mock('@stripe/stripe-js/pure', () => ({ loadStripe: loadStripeMock }))

import { getStripe, isClientSecret, isPublishableKey } from '@/lib/stripe'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('getStripe', () => {
  it('loads Stripe.js once per publishable key, and hands every caller the same promise', async () => {
    const first = getStripe('pk_test_once')
    const again = getStripe('pk_test_once')

    expect(again).toBe(first)
    await expect(first).resolves.toEqual({ key: 'pk_test_once' })
    expect(loadStripeMock.mock.calls.filter(([key]) => key === 'pk_test_once')).toHaveLength(1)
    expect(getStripe('pk_live_other')).not.toBe(first)
  })

  it('loads nothing for no key, an empty key, or a secret key', () => {
    loadStripeMock.mockClear()

    expect(getStripe(null)).toBeNull()
    expect(getStripe(undefined)).toBeNull()
    expect(getStripe('')).toBeNull()
    expect(getStripe('sk_test_short')).toBeNull()
    expect(getStripe('rk_live_short')).toBeNull()
    expect(loadStripeMock).not.toHaveBeenCalled()
  })

  it('loads nothing during a server render', () => {
    loadStripeMock.mockClear()
    vi.stubGlobal('window', undefined)

    expect(getStripe('pk_test_server')).toBeNull()
    expect(loadStripeMock).not.toHaveBeenCalled()
  })

  it('forgets a failed load, so the next attempt loads again', async () => {
    loadStripeMock.mockRejectedValueOnce(new Error('offline'))

    await expect(getStripe('pk_test_flaky')).rejects.toThrow('offline')
    await expect(getStripe('pk_test_flaky')).resolves.toEqual({ key: 'pk_test_flaky' })
    expect(loadStripeMock.mock.calls.filter(([key]) => key === 'pk_test_flaky')).toHaveLength(2)
  })
})

describe('the Stripe value checks', () => {
  it('knows a publishable key', () => {
    expect(isPublishableKey('pk_test_abc123')).toBe(true)
    expect(isPublishableKey('pk_live_abc123')).toBe(true)
    expect(isPublishableKey('sk_live_abc123')).toBe(false)
    expect(isPublishableKey('pk_test_')).toBe(false)
    expect(isPublishableKey(42)).toBe(false)
  })

  it('knows each kind of client secret', () => {
    expect(isClientSecret('cs_test_a1_secret_b2', 'cs_')).toBe(true)
    expect(isClientSecret('seti_a1_secret_b2', 'seti_')).toBe(true)
    expect(isClientSecret('pi_a1_secret_b2', 'pi_')).toBe(true)
    expect(isClientSecret('pi_a1_secret_b2', 'cs_')).toBe(false)
    expect(isClientSecret('cs_test_a1', 'cs_')).toBe(false)
    expect(isClientSecret('https://checkout.stripe.com/c/pay/cs_test_a1_secret_b2', 'cs_')).toBe(false)
    expect(isClientSecret(null, 'cs_')).toBe(false)
  })
})
