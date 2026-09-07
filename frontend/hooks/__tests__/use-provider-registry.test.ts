/**
 * PRD-236 S0.6 — the provider registry hook degrades honestly.
 *
 * The settings pages and the onboarding card render the backend registry; when
 * that call is unavailable (stubbed client, network failure, malformed body)
 * they must render exactly the list they rendered before PRD-236.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

// vi.mock is hoisted above every other statement, so the mock fn must be
// hoisted with it (a plain `const` here is a temporal-dead-zone error in CI).
const { getMock } = vi.hoisted(() => ({ getMock: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { get: getMock } }))

import {
  STATIC_PROVIDER_FALLBACK,
  chatProviders,
  hostsVendorModels,
  keyPlaceholder,
  loadProviderRegistry,
  providerNotes,
  providersForByok,
  providersForPlatformKeys,
  resetProviderRegistryCache,
} from '@/hooks/use-provider-registry'

const registryFromBackend = {
  edition: 'saas' as const,
  providers: [
    ...STATIC_PROVIDER_FALLBACK.providers.slice(0, 4),
    {
      slug: 'nvidia',
      label: 'NVIDIA',
      kind: 'hosted_open' as const,
      chat: true,
      embeddings: false,
      byok: true,
      platform_key: false,
      hosts_vendor_models: true,
      free: true,
      key_placeholder: 'nvapi-…',
      docs_url: 'https://build.nvidia.com/',
      terms_note: 'NVIDIA trial: testing and evaluation only.',
      rate_limit_note: 'About 40 requests per minute.',
    },
  ],
}

afterEach(() => {
  resetProviderRegistryCache()
  getMock.mockReset()
})

describe('loadProviderRegistry', () => {
  it('returns the backend registry when the call succeeds, fetching once', async () => {
    getMock.mockResolvedValue(registryFromBackend)
    const first = await loadProviderRegistry()
    const second = await loadProviderRegistry()
    expect(first).toBe(second)
    expect(getMock).toHaveBeenCalledTimes(1)
    expect(getMock).toHaveBeenCalledWith('/api/keys/providers')
    expect(first.providers.map((p) => p.slug)).toContain('nvidia')
  })

  it('falls back to the static list when the call fails', async () => {
    getMock.mockRejectedValue(new Error('offline'))
    const registry = await loadProviderRegistry()
    expect(registry).toBe(STATIC_PROVIDER_FALLBACK)
    expect(registry.providers.map((p) => p.slug)).toEqual([
      'openai', 'anthropic', 'google', 'openrouter', 'deepseek',
      'azure', 'bedrock', 'grok', 'cohere', 'huggingface',
    ])
  })

  it('falls back when the body is not a registry', async () => {
    getMock.mockResolvedValue({ nope: true })
    expect(await loadProviderRegistry()).toBe(STATIC_PROVIDER_FALLBACK)
  })
})

describe('helpers', () => {
  it('platform-key list drops byok_only providers, BYOK list keeps them', () => {
    expect(providersForPlatformKeys(registryFromBackend).map((p) => p.slug)).not.toContain('nvidia')
    expect(providersForByok(registryFromBackend).map((p) => p.slug)).toContain('nvidia')
  })

  it('knows which providers host vendor-prefixed models', () => {
    expect(hostsVendorModels(registryFromBackend, 'openrouter')).toBe(true)
    expect(hostsVendorModels(registryFromBackend, 'nvidia')).toBe(true)
    expect(hostsVendorModels(registryFromBackend, 'openai')).toBe(false)
    expect(hostsVendorModels(STATIC_PROVIDER_FALLBACK, 'openrouter')).toBe(true)
  })

  it('surfaces the trial and rate-limit notes and the key placeholder', () => {
    expect(providerNotes(registryFromBackend, 'nvidia')).toHaveLength(2)
    expect(providerNotes(registryFromBackend, 'openai')).toEqual([])
    expect(keyPlaceholder(registryFromBackend, 'nvidia')).toBe('nvapi-…')
    expect(keyPlaceholder(registryFromBackend, 'unknown')).toBe('Paste your API key')
  })

  it('chat providers exclude key-only ones', () => {
    expect(chatProviders(STATIC_PROVIDER_FALLBACK).map((p) => p.slug)).not.toContain('cohere')
  })
})
