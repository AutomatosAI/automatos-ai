/**
 * PRD-255 Wave 1, US-002 — each tone word can carry a one-line meaning. The kit answers
 * `voice.tone` as `[{word, meaning}]` (an older backend: plain strings); the form reads
 * either, and editing the words keeps the meaning of every word that stays, so a Save
 * never wipes the meanings the owner (or the Brand designer) gave.
 */
import { describe, it, expect, vi } from 'vitest'

import { toneWordsFrom, withToneWords } from '@/components/documents/blocks/BrandKitSocial'
import { withD5Fields } from '../use-brand-kit-form'
import type { BrandKit } from '@/components/documents/blocks/types'

vi.mock('@/lib/api-client', () => ({ apiClient: { get: vi.fn(), put: vi.fn(), post: vi.fn(), delete: vi.fn() } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

const WARM = { word: 'warm', meaning: 'friendly, never gushing' }

describe('tone words with meanings', () => {
  it('reads plain strings as words with no meaning, and keeps a meaning it is given', () => {
    expect(toneWordsFrom(['plain', WARM])).toEqual([{ word: 'plain', meaning: '' }, WARM])
    expect(toneWordsFrom(undefined)).toEqual([])
  })

  it('keeps the meaning of each word that stays, in any case, when the words are edited', () => {
    const current = [WARM, { word: 'plain', meaning: 'short words' }, { word: 'local', meaning: '' }]
    expect(withToneWords(current, ['Warm', 'local', 'curious'])).toEqual([
      { word: 'Warm', meaning: 'friendly, never gushing' },
      { word: 'local', meaning: '' },
      { word: 'curious', meaning: '' },
    ])
  })

  it('fills the form from a kit whose tone words are plain strings or objects', () => {
    const base = {
      name: 'Acme', tagline: '', logo_url: '', logo_path: '',
      primary_color: '#1a1a2e', secondary_color: '#16213e', accent_color: '#0f3460', text_color: '#1a1a2e',
      font_family: 'Inter, sans-serif',
      company: { name: '', address: '', email: '', phone: '', website: '' },
    }
    const old = withD5Fields({ ...base, voice: { tone: ['warm', 'plain', 'local'], banned_phrases: [] } } as unknown as BrandKit)
    expect(old.voice.tone.map((t) => t.meaning)).toEqual(['', '', ''])
    const v2 = withD5Fields({ ...base, voice: { tone: [WARM], banned_phrases: [] } } as unknown as BrandKit)
    expect(v2.voice.tone).toEqual([WARM])
  })
})
