/**
 * PRD-259 S-004b test support (not a test file): installed-skin items built from the shared
 * S-001 corpus, and the providers' tone-lock wiring (ThemeProvider's forcedTheme fed by
 * UiStyleProvider's onToneLock), as components/providers.tsx wires it.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { useState, type ReactNode } from 'react'
import { ThemeProvider } from 'next-themes'

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import type { SkinTone } from '@/lib/skins/schema'
import type { BaseStyle, UiStyle } from '@/lib/ui-style'

export const SKIN_ID = '0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90'
export const OTHER_SKIN_ID = '11111111-2222-4333-8444-555555555555'
export const THEME_STORAGE_KEY = 'automatos-theme'

export const CASES = {
  twoTone: "Studio's own light and dark tokens, as a skin",
  darkOnly: "a dark-only skin (Studio Dark's tokens)",
  classicLightOnly: 'the minimal skin: required fields only',
} as const

interface CorpusCase {
  name: string
  skin: Record<string, unknown>
}

const corpus = JSON.parse(
  readFileSync(path.resolve(__dirname, '../../../orchestrator/tests/fixtures/skins/cases.json'), 'utf8'),
) as { cases: CorpusCase[] }

/** One item as GET /api/skins/installed returns it (records.installed_item), from a corpus case. */
export function installedItem(caseName: string, id: string, overrides: Record<string, unknown> = {}) {
  const found = corpus.cases.find((testCase) => testCase.name === caseName)
  if (!found) throw new Error(`no corpus case "${caseName}"`)
  const { slug, name, version, ...skin } = found.skin
  const payload = ['base', 'tones', 'tokens', 'fonts', 'shape', 'mascot', 'preview']
    .map((field) => [field, skin[field] ?? null] as const)
  return { id, slug, name, version, ...Object.fromEntries(payload), ownerType: 'workspace', ...overrides }
}

interface HarnessProps {
  initialStyle?: UiStyle
  initialSkinBase?: BaseStyle
  children: ReactNode
}

/** ThemeProvider + UiStyleProvider with the tone lock lifted, as Providers does it. */
export function SkinHarness({ initialStyle, initialSkinBase, children }: HarnessProps) {
  const [lockedTone, setLockedTone] = useState<SkinTone | undefined>(undefined)
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem
      themes={['light', 'dark']}
      storageKey={THEME_STORAGE_KEY}
      forcedTheme={lockedTone}
    >
      <UiStyleProvider initialStyle={initialStyle} initialSkinBase={initialSkinBase} onToneLock={setLockedTone}>
        {children}
      </UiStyleProvider>
    </ThemeProvider>
  )
}

/** Reads and drives the style context from a test. */
export function StyleProbe() {
  const { style, setStyle, isStudio, base, installedSkins } = useUiStyle()
  return (
    <div>
      <span data-testid="style">{style}</span>
      <span data-testid="studio">{String(isStudio)}</span>
      <span data-testid="base">{base}</span>
      <span data-testid="skins">{installedSkins.status}:{installedSkins.skins.map((skin) => skin.name).join('|')}</span>
      <button onClick={() => { void installedSkins.reload() }}>load skins</button>
      <button onClick={() => setStyle(`skin:${SKIN_ID}`)}>go skin</button>
      <button onClick={() => setStyle(`skin:${OTHER_SKIN_ID}`)}>go other skin</button>
      <button onClick={() => setStyle('studio')}>go studio</button>
      <button onClick={() => setStyle('classic')}>go classic</button>
    </div>
  )
}
