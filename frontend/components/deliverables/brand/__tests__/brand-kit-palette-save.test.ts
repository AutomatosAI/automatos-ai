/**
 * PRD-255 Wave 1, US-001 — GET /api/documents/brand-kit answers every effective colour role,
 * the derived ones too, and says per role whether it is set or derived. Save (kitToSave) sends
 * only the set roles: a derived role sent back would be stored as set, and would stop following
 * the kit's colours. palette_source is the server's answer, never sent.
 */
import { describe, it, expect, vi } from 'vitest'

import { kitToSave } from '../use-brand-kit-form'
import type { BrandKit } from '@/components/documents/blocks/types'

vi.mock('@/lib/api-client', () => ({ apiClient: { get: vi.fn(), put: vi.fn(), post: vi.fn(), delete: vi.fn() } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

function kit(extra: Partial<BrandKit>): BrandKit {
  return {
    name: 'Harbourline', tagline: '', logo_url: '', logo_path: '',
    primary_color: '#1e3a5f', secondary_color: '#c26a2e', accent_color: '#0f3460', text_color: '#1a1a2e',
    font_family: 'Inter, sans-serif',
    company: { name: '', address: '', email: '', phone: '', website: '' },
    heading_font: '', font_files: [], logo_mark_url: '', logo_mark_path: '',
    voice: { tone: [], banned_phrases: [], sign_off: '' },
    ...extra,
  }
}

describe('kitToSave', () => {
  it('sends only the roles the owner set, and never palette_source', () => {
    const saved = kitToSave(kit({
      palette: { ink: '#1a1a2e', heading: '#0b0b14', paper: '#ffffff', accent: '#1e3a5f' },
      palette_source: { ink: 'derived', heading: 'derived', paper: 'set', accent: 'set' },
      accent_use: 'bold',
    }))
    expect(saved.palette).toEqual({ paper: '#ffffff', accent: '#1e3a5f' })
    expect(saved).not.toHaveProperty('palette_source')
    expect(saved.accent_use).toBe('bold')
  })

  it('sends no roles when every role is derived', () => {
    const saved = kitToSave(kit({
      palette: { ink: '#1a1a2e', paper: '#ffffff' },
      palette_source: { ink: 'derived', paper: 'derived' },
    }))
    expect(saved.palette).toEqual({})
  })

  it('leaves a kit from a backend without roles as it was', () => {
    const saved = kitToSave(kit({}))
    expect(saved).not.toHaveProperty('palette')
    expect(saved.name).toBe('Harbourline')
  })
})
