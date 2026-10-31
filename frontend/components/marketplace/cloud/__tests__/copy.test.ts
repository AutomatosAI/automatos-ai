/**
 * PRD-270 P270W2-RVW-6 (§5.1, §5.6, K19, K30) — the words of Sync with Automatos Cloud after W2. The status
 * line names the agents, Playbooks, packages and plugins this install holds after its themes, leaving out a
 * type it holds none of; the Settings lede says how each kind is trusted: themes signed by an Automatos key,
 * the other items Reviewed on Automatos Cloud and shown before they are added.
 */
import { describe, expect, it } from 'vitest'

import type { CloudSyncStatus } from '@/lib/marketplace/cloud-sync'

import { CLOUD_LEDE, NOT_SYNCED_YET, itemsHeldLine, syncedLine } from '../copy'

const NOW = Date.parse('2026-10-31T12:00:00Z')
const TWO_MINUTES_AGO = new Date(NOW - 2 * 60_000).toISOString()
const NO_ITEMS: CloudSyncStatus['items'] = { agent: 0, playbook: 0, package: 0, plugin: 0 }

function line(themes: number, items: Partial<CloudSyncStatus['items']>): string {
  return syncedLine({ lastSuccessAt: TWO_MINUTES_AGO, themes, items: { ...NO_ITEMS, ...items } }, NOW)
}

describe('the status line beside Sync with Automatos Cloud', () => {
  it('names each item type held after the themes, in the order agents, Playbooks, packages, plugins', () => {
    expect(line(3, { agent: 2, playbook: 1, package: 4, plugin: 1 })).toBe(
      'Last synced 2m ago · 3 themes · 2 agents · 1 Playbook · 4 packages · 1 plugin',
    )
    expect(line(1, { agent: 1, playbook: 2, package: 1, plugin: 3 })).toBe(
      'Last synced 2m ago · 1 theme · 1 agent · 2 Playbooks · 1 package · 3 plugins',
    )
  })

  it('leaves out a type this install holds none of', () => {
    expect(line(3, { playbook: 2 })).toBe('Last synced 2m ago · 3 themes · 2 Playbooks')
    expect(line(0, { plugin: 1 })).toBe('Last synced 2m ago · 0 themes · 1 plugin')
    expect(line(3, {})).toBe('Last synced 2m ago · 3 themes')
  })

  it('says not synced yet before any Sync, whatever is held', () => {
    expect(syncedLine({ lastSuccessAt: null, themes: 0, items: { ...NO_ITEMS, agent: 2 } }, NOW)).toBe(NOT_SYNCED_YET)
  })
})

describe("Settings' items held here", () => {
  it('names each item type held, leaving out a type with none, or says none', () => {
    expect(itemsHeldLine({ agent: 2, playbook: 1, package: 0, plugin: 1 })).toBe('2 agents, 1 Playbook, 1 plugin')
    expect(itemsHeldLine({ ...NO_ITEMS, package: 3 })).toBe('3 packages')
    expect(itemsHeldLine(NO_ITEMS)).toBe('none')
  })
})

describe('the Settings lede', () => {
  it('says how each kind is trusted: themes signed, the other items Reviewed on Automatos Cloud and shown first', () => {
    expect(CLOUD_LEDE).toContain('Reviewed on Automatos Cloud')
    expect(CLOUD_LEDE).toContain('theme versions an Automatos key signed')
    expect(CLOUD_LEDE).toContain('agents, Playbooks, packages and plugins')
    expect(CLOUD_LEDE).toContain('before it is added')
    expect(CLOUD_LEDE).toContain('only when someone presses it')
  })

  it('no longer says every version it lists is signed, and never calls an item Official', () => {
    expect(CLOUD_LEDE).not.toContain('lists only versions an Automatos key signed')
    expect(CLOUD_LEDE).not.toMatch(/official/i)
  })
})
