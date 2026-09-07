/**
 * Provider registry hook (PRD-236 S0.6)
 * =====================================
 *
 * The backend's provider registry (`core/llm/providers.py`) as the UI sees it:
 * labels, key placeholders, docs links, the NVIDIA trial and rate-limit notes,
 * and per-edition flags. Every provider list in Settings and onboarding renders
 * this instead of its own hardcoded array.
 *
 * Deliberately NOT a react-query hook: the onboarding power-up card renders
 * outside a QueryClientProvider in tests and must keep working when the
 * registry call is unavailable. One fetch per page load is shared through a
 * module-level promise; any failure (or a missing `apiClient.get` in a stub)
 * degrades to STATIC_PROVIDER_FALLBACK — today's list, unchanged.
 */

import { useEffect, useState } from 'react'
import { apiClient } from '@/lib/api-client'

export type ProviderKind = 'direct' | 'aggregator' | 'hosted_open'

export interface ProviderSpec {
  slug: string
  label: string
  kind: ProviderKind
  chat: boolean
  embeddings: boolean
  /** May a workspace add its own key for this provider. */
  byok: boolean
  /** May an operator hold a platform-level key (false for byok_only providers in saas). */
  platform_key: boolean
  /** Accepts vendor-prefixed ids like "moonshotai/kimi-k3" (openrouter, nvidia). */
  hosts_vendor_models: boolean
  /** The provider does not bill for calls (NVIDIA trial). */
  free: boolean
  key_placeholder: string
  docs_url: string | null
  terms_note: string | null
  rate_limit_note: string | null
}

export interface ProviderRegistry {
  edition: 'local' | 'saas' | null
  providers: ProviderSpec[]
}

export const PROVIDER_REGISTRY_PATH = '/api/keys/providers'

function staticSpec(
  slug: string,
  label: string,
  extra: Partial<ProviderSpec> = {},
): ProviderSpec {
  return {
    slug,
    label,
    kind: 'direct',
    chat: true,
    embeddings: false,
    byok: true,
    platform_key: true,
    hosts_vendor_models: false,
    free: false,
    key_placeholder: 'Paste your API key',
    docs_url: null,
    terms_note: null,
    rate_limit_note: null,
    ...extra,
  }
}

/** Today's list, byte-for-byte the options the settings pages rendered before PRD-236. */
export const STATIC_PROVIDER_FALLBACK: ProviderRegistry = {
  edition: null,
  providers: [
    staticSpec('openai', 'OpenAI', { key_placeholder: 'sk-…' }),
    staticSpec('anthropic', 'Anthropic', { key_placeholder: 'sk-ant-…' }),
    staticSpec('google', 'Google'),
    staticSpec('openrouter', 'OpenRouter', {
      kind: 'aggregator',
      hosts_vendor_models: true,
      key_placeholder: 'sk-or-…',
    }),
    staticSpec('deepseek', 'DeepSeek'),
    staticSpec('azure', 'Azure OpenAI'),
    staticSpec('bedrock', 'AWS Bedrock'),
    staticSpec('grok', 'Grok / xAI'),
    staticSpec('cohere', 'Cohere', { chat: false }),
    staticSpec('huggingface', 'HuggingFace'),
  ],
}

let registryPromise: Promise<ProviderRegistry> | null = null

function isRegistry(value: unknown): value is ProviderRegistry {
  return (
    !!value &&
    typeof value === 'object' &&
    Array.isArray((value as ProviderRegistry).providers) &&
    (value as ProviderRegistry).providers.length > 0
  )
}

/** Fetch once per page load; never throws — resolves to the static fallback on any failure. */
export function loadProviderRegistry(): Promise<ProviderRegistry> {
  if (!registryPromise) {
    registryPromise = (async () => {
      try {
        const getter = (apiClient as { get?: (path: string) => Promise<unknown> } | undefined)?.get
        if (typeof getter !== 'function') return STATIC_PROVIDER_FALLBACK
        const data = await getter.call(apiClient, PROVIDER_REGISTRY_PATH)
        return isRegistry(data) ? data : STATIC_PROVIDER_FALLBACK
      } catch {
        return STATIC_PROVIDER_FALLBACK
      }
    })()
  }
  return registryPromise
}

/** Test seam: forget the cached registry so the next call fetches again. */
export function resetProviderRegistryCache() {
  registryPromise = null
}

export function useProviderRegistry(): ProviderRegistry {
  const [registry, setRegistry] = useState<ProviderRegistry>(STATIC_PROVIDER_FALLBACK)

  useEffect(() => {
    let cancelled = false
    loadProviderRegistry().then((data) => {
      if (!cancelled) setRegistry(data)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return registry
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function providersForByok(registry: ProviderRegistry): ProviderSpec[] {
  return registry.providers.filter((p) => p.byok)
}

export function providersForPlatformKeys(registry: ProviderRegistry): ProviderSpec[] {
  return registry.providers.filter((p) => p.platform_key)
}

export function chatProviders(registry: ProviderRegistry): ProviderSpec[] {
  return registry.providers.filter((p) => p.chat)
}

export function findProvider(registry: ProviderRegistry, slug: string | null | undefined): ProviderSpec | undefined {
  if (!slug) return undefined
  return registry.providers.find((p) => p.slug === slug)
}

export function providerLabel(registry: ProviderRegistry, slug: string | null | undefined): string {
  return findProvider(registry, slug)?.label ?? (slug ?? '')
}

export function hostsVendorModels(registry: ProviderRegistry, slug: string | null | undefined): boolean {
  return findProvider(registry, slug)?.hosts_vendor_models ?? false
}

export function keyPlaceholder(registry: ProviderRegistry, slug: string | null | undefined): string {
  return findProvider(registry, slug)?.key_placeholder ?? 'Paste your API key'
}

/** The trial / rate-limit text a user must see before saving a key for this provider. */
export function providerNotes(registry: ProviderRegistry, slug: string | null | undefined): string[] {
  const spec = findProvider(registry, slug)
  if (!spec) return []
  return [spec.terms_note, spec.rate_limit_note].filter((n): n is string => !!n)
}
