/**
 * PRD-244 (two styles, two tones) — the Classic Command Centre carries the
 * Munder Difflin surfaces (Watchlist · Questions · Governance) in its own
 * style, mounting the same tab bodies the Studio shell mounts.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const src = readFileSync(path.resolve(__dirname, '..', 'activity-page.tsx'), 'utf8')

describe('Classic Command Centre tabs', () => {
  it('declares the three tabs so the bell deep-links (?tab=questions, ?tab=watchlist) land', () => {
    for (const v of ['watchlist', 'questions', 'governance']) {
      expect(src).toContain(`{ value: '${v}',`)
      expect(src).toContain(`<TabsContent value="${v}">`)
    }
  })

  it('mounts the shared tab bodies, Governance in its classic variant', () => {
    expect(src).toContain('<WatchlistTab />')
    expect(src).toContain('<QuestionsTab />')
    expect(src).toContain('<GovernanceTab variant="classic" />')
  })

  it('the Questions badge is the same read as the shell: open question-kind grants', () => {
    expect(src).toContain("from '@/hooks/use-approval-grants'")
    expect(src).toContain('questions?.grants?.length ?? 0')
  })

  it('the tab bodies carry no Studio class of their own (they must render in either style)', () => {
    const dir = path.resolve(__dirname, '..', '..', 'command-center')
    for (const f of ['questions-tab.tsx', 'watchlist-tab.tsx']) {
      expect(readFileSync(path.join(dir, f), 'utf8')).not.toMatch(/className="[^"]*\bcc-/)
    }
  })
})
