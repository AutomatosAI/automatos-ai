/**
 * PRD-244 review batch 3 — gates for the two root causes Gerard's pass found:
 * the Studio tab strip vanishing on tall pages, and markdown previews rendering
 * as a wall of text.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '..', '..', '..')
const read = (rel: string) => readFileSync(path.join(ROOT, rel), 'utf8')
const css = read('app/globals.css')

const MARKDOWN_SURFACES = [
  'components/widgets/FileWidget/FilePreview.tsx',
  'components/deliverables/blog-editor.tsx',
  'components/agents/skills/skill-detail-modal.tsx',
  'components/missions/mission-results-panel.tsx',
  'components/chatbot/text-artifact.tsx',
  'components/chatbot/sheet-artifact.tsx',
]

describe('Studio page chrome', () => {
  it('the tab strip never absorbs the page overflow', () => {
    const block = css.slice(css.indexOf(':is(.studio, .cc-page) .cc-tabs {'), css.indexOf('}', css.indexOf(':is(.studio, .cc-page) .cc-tabs {')))
    expect(block).toContain('overflow-x: auto')
    expect(block, 'overflow-x zeroes the automatic minimum size — the strip must not shrink').toContain('flex-shrink: 0')
    expect(css).toMatch(/\.cc-headrow \{[^}]*flex-shrink: 0/)
  })
})

describe('Markdown typography', () => {
  it('the app carries its own document typography — @tailwindcss/typography is not installed', () => {
    const pkg = JSON.parse(read('package.json'))
    const deps = { ...pkg.dependencies, ...pkg.devDependencies }
    expect(deps['@tailwindcss/typography'], 'if this is ever added, reconcile it with .md-view').toBeUndefined()
    for (const rule of ['.md-view {', '.md-view h1 {', '.md-view ul {', '.md-view table {', '.md-view code {', '.md-view blockquote {', '.md-view-compact {']) {
      expect(css, rule).toContain(rule)
    }
    expect(css, 'Studio heads read in the paper serif').toContain('.studio .md-view h1')
  })

  it('every markdown preview surface renders through the shared view, with no inert prose class left', () => {
    for (const rel of MARKDOWN_SURFACES) {
      const src = read(rel)
      expect(src, rel).toContain('<MarkdownView')
      expect(src, rel).toContain("from '@/components/shared/markdown-view'")
      expect(src.match(/className="[^"]*\bprose\b/), rel).toBeNull()
    }
  })

  it('the question ask body uses the same typography', () => {
    expect(read('components/command-center/questions-tab.tsx')).toContain('className="md-view md-view-compact"')
  })
})
