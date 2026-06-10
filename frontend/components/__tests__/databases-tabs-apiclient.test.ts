import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

// PRD-154 S12 — the three Databases sub-tabs used raw fetch('/api/…') with no
// baseUrl and no auth header, so they 404'd in split-origin deploys (frontend
// origin ≠ API origin) and the Training tab never loaded. Routing every call
// through apiClient injects the configured baseUrl + Clerk/workspace headers.
//
// Deterministic proxy for the browser AC (six Databases sub-tabs load in
// split-origin dev): no raw /api fetch survives, and apiClient is the client.

const ROOT = path.resolve(__dirname, '..', '..')

const FILES = [
  'components/context/DatabaseQueryAnalytics.tsx',
  'components/knowledge/SemanticLayerBuilder.tsx',
  'components/knowledge/QueryTemplatesGrid.tsx',
]

describe('S12 Databases tabs — apiClient, not raw fetch', () => {
  for (const rel of FILES) {
    const src = readFileSync(path.join(ROOT, rel), 'utf8')

    it(`${rel} imports apiClient`, () => {
      expect(src).toMatch(/import\s*\{\s*apiClient\s*\}\s*from\s*['"]@\/lib\/api-client['"]/)
    })

    it(`${rel} has no raw fetch('/api…') call`, () => {
      // Backtick or quote — both forms 404 in split-origin without a baseUrl.
      expect(src).not.toMatch(/fetch\(\s*['"`]\/api/)
    })
  }
})
