/** PRD-255 US-007 — a brand kit as GET answers it on a v2 backend, for the design-system section tests. */
import type { BrandKit } from '@/components/documents/blocks/types'

export function designKit(extra: Partial<BrandKit> = {}): BrandKit {
  return {
    name: 'Harbourline', tagline: 'Coffee by the water', logo_url: '', logo_path: '',
    primary_color: '#c2410c', secondary_color: '#64748b', accent_color: '#c2410c', text_color: '#1a1a2e',
    font_family: 'Inter, sans-serif',
    company: { name: 'Harbourline Ltd', address: '', email: '', phone: '', website: 'harbourline.co' },
    heading_font: '"Brand Serif", serif', font_files: [], logo_mark_url: '', logo_mark_path: '',
    voice: { tone: [], banned_phrases: [], sign_off: '' },
    palette: {
      ink: '#1a1a2e', heading: '#111118', paper: '#fbfaf7', surface: '#f3f1ec', surface_2: '#ebe8e1',
      accent: '#9a3412', accent_2: '#475569', muted: '#5b6170', rule: '#dcd8cf',
    },
    palette_source: {
      ink: 'derived', heading: 'derived', paper: 'derived', surface: 'derived', surface_2: 'derived',
      accent: 'set', accent_2: 'derived', muted: 'derived', rule: 'derived',
    },
    accent_use: 'sparing',
    type_scale: {
      display: { size_pt: 32, line_pt: 38, weight: 700 },
      h1: { size_pt: 22, line_pt: 28, weight: 700 },
      h2: { size_pt: 16, line_pt: 22, weight: 600 },
      h3: { size_pt: 13, line_pt: 18, weight: 600 },
      body: { size_pt: 10.5, line_pt: 15, weight: 400 },
      small: { size_pt: 9, line_pt: 13, weight: 400 },
      caption: { size_pt: 8, line_pt: 11, weight: 400 },
    },
    spacing_unit_pt: 4,
    page_margin_mm: 18,
    logo_rules: { letterhead_mm: 16, clear_space: 0.5, min_mm: 8 },
    logo_dark_path: '',
    logo_mono_path: '',
    currency: '',
    date_style: 'd MMMM yyyy',
    ...extra,
  }
}
