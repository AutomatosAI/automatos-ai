/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the public items in the member cards' shapes:
 * what the public item says is mapped, what it doesn't say is left empty, never made up.
 */
import { describe, expect, it } from 'vitest'

import { publicItem } from '@/lib/marketplace/public-items'
import type { PublicMarketplaceType } from '@/lib/marketplace/public-items'
import type { PublicEntry } from '@/lib/marketplace/source/public-reads'
import { publicEntry } from '@/lib/marketplace/__tests__/public-fixtures'

import { routeKeyOf } from '../../llm-model-card'
import { appOf, llmOf, pluginOf, skillOf } from '../public-capability-shapes'
import { featuredItemOf, marketplaceItemOf } from '../public-shapes'

function entry(type: PublicMarketplaceType, overrides: Record<string, unknown>): PublicEntry {
  const raw = publicEntry(overrides)
  const item = publicItem(raw, type)
  if (!item) throw new Error('fixture is not a public item')
  return { item, raw }
}

describe('agents, Playbooks and Featured', () => {
  it('maps an agent by its catalogue id, its required apps as its tools, and drops a key that isn’t an id', () => {
    const agent = marketplaceItemOf(entry('agents', { key: '7', requiredApps: ['notion', 'gmail'] }).item, 'agent')
    expect(agent).toMatchObject({ id: 7, type: 'agent', icon: '🔎', category: 'Research', tags: ['research', 'web'] })
    expect(agent?.metadata).toEqual({ tool_names: ['NOTION', 'GMAIL'], tool_icons: [] })
    expect(marketplaceItemOf(entry('agents', { key: 'research-assistant' }).item, 'agent')).toBeNull()
  })

  it('reads a Featured item’s type, version and install count; an unknown type is no item', () => {
    const raw = { ...publicEntry({ key: '12' }), type: 'playbooks', version: '3.1.0', installCount: 9 }
    expect(featuredItemOf(raw)).toMatchObject({ id: 12, type: 'recipe', version: '3.1.0', install_count: 9 })
    expect(featuredItemOf({ ...raw, type: 'themes' })).toBeNull()
    expect(featuredItemOf({ ...raw, installCount: -3, version: 7 })).toMatchObject({ install_count: 0, version: '' })
  })

  it('keeps an icon only as the public item checks it: an https logo or a short text', () => {
    expect(marketplaceItemOf(entry('agents', { key: '1', icon: 'javascript:alert(1)' }).item, 'agent')?.icon).toBeUndefined()
  })
})

describe('applications, plugins, skills and LLM routes', () => {
  it('an application: its slug, name, https logo, categories and counts; no auth schemes or connection', () => {
    const app = appOf(entry('applications', { key: 'notion', name: 'Notion', icon: 'https://logos.example/notion', tags: ['docs'], actionCount: 12, triggerCount: -1 }))
    expect(app).toEqual({
      name: 'notion', display_name: 'Notion', description: 'Finds, reads and summarises sources.',
      logo_url: 'https://logos.example/notion', categories: ['docs'], is_connected: false,
      action_count: 12, trigger_count: 0, auth_schemes: [], triggers: [],
    })
    expect(appOf(entry('applications', { icon: '🔎' })).logo_url).toBeUndefined()
  })

  it('a plugin: its id, version, counts and publisher; no enables, security or approval state', () => {
    const plugin = pluginOf(entry('plugins', { key: 'p-1', version: '1.2.0', skillsCount: 3, publisher: 'Ada', category: 'Engineering' }))
    expect(plugin).toMatchObject({
      id: 'p-1', version: '1.2.0', skills_count: 3, commands_count: 0, original_author: 'Ada',
      category_name: 'Engineering', enable_count: 0, is_featured: false, security_status: null, approval_status: null,
    })
  })

  it('a skill: its name, version and category; no token estimate or source', () => {
    expect(skillOf(entry('skills', { key: 'pdf', name: 'PDF', version: '1.0.0' }), 4)).toMatchObject({
      id: 4, name: 'PDF', skill_version: '1.0.0', estimated_tokens: 0, skill_source: null, is_enabled: false,
    })
  })

  it('an LLM route: its key’s provider and model id, vendor, context window, tier and ratings', () => {
    const model = llmOf(
      entry('llms', {
        key: 'openrouter:moonshotai/kimi-k3', name: 'Kimi K3', publisher: 'OpenRouter', provider: 'openrouter',
        vendor: 'moonshotai', contextWindow: 256000, priceTier: 'free',
        capabilities: [{ name: 'coding', rating: 'good' }, { name: 'bad' }],
      }),
      0,
    )
    expect(model).toMatchObject({
      serving_provider: 'openrouter', serving_provider_label: 'OpenRouter', provider: 'moonshotai',
      model_id: 'moonshotai/kimi-k3', context_window: 256000, is_free: true, price_tier: 'free',
      capabilities: { coding: 'good' }, is_installed: false,
    })
    expect(model.key_available).toBeUndefined() // no "add a key" badge: a workspace's keys aren't public
    expect(routeKeyOf(model)).toBe('openrouter:moonshotai/kimi-k3')
  })
})
