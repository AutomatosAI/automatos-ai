/**
 * PRD-246 US-005 — Agents, Deliverables and Tools on a phone.
 *
 * The three bespoke Studio pages share their compact form with the rest of
 * the `.cc-page` surfaces (the 16px gutter, the matched tab bleed, `.cc-stats`
 * 2-up — all US-002's rules on the same families). What is specific to this
 * story: the tab strips use the ONE scroll-into-view hook rather than three
 * copies, and the toolbars wrap with the search on its own line.
 *
 * Their card grids are Tailwind's own `grid-cols-1 md:grid-cols-2 …` — one
 * column below 768 already — so this asserts that rather than adding a
 * second rule for it. A regression to `sm:grid-cols-2` fails here.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const width = vi.hoisted(() => ({ compact: false }))
const nav = vi.hoisted(() => ({ search: '', replace: vi.fn(), push: vi.fn() }))

vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => width.compact,
  useIsTabletOrBelow: () => width.compact,
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: nav.push }),
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('@/hooks/use-agent-api', () => ({
  useAgents: () => ({ data: [], isLoading: false, refetch: vi.fn(), error: null }),
  useAgentStats: () => ({ data: undefined }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ canEdit: true, workspace: { id: 'ws-1' }, isLoading: false }),
}))
vi.mock('@/hooks/use-view-mode', () => ({ useViewMode: () => ['grid', vi.fn()] }))
vi.mock('@/components/agents/agent-roster', () => ({ AgentRoster: () => <div /> }))
vi.mock('@/components/agents/fleet-tab', () => ({ FleetTab: () => <div /> }))
vi.mock('@/components/agents/org-chart-tab', () => ({ OrgChartTab: () => <div /> }))
vi.mock('@/components/agents/agent-configuration', () => ({ AgentConfiguration: () => <div /> }))
vi.mock('@/components/agents/skills/workspace-skills-tab', () => ({ WorkspaceSkillsTab: () => <div /> }))
vi.mock('@/components/agents/create-agent-modal', () => ({ CreateAgentModal: () => null }))
vi.mock('@/components/agents/agent-details-modal', () => ({ AgentDetailsModal: () => null }))
vi.mock('@/components/deliverables/outputs-feed', () => ({ OutputsFeed: () => <div /> }))
vi.mock('@/components/deliverables/deliverables-blogs', () => ({ DeliverablesBlog: () => <div /> }))
vi.mock('@/components/documents/blocks/TemplateStudio', () => ({ TemplateStudio: () => <div /> }))
vi.mock('@/components/workspace/gallery-view', () => ({ GalleryView: () => <div /> }))

import { AgentManagementStudio } from '../agent-management-studio'
import { DeliverablesStudio } from '@/components/deliverables/studio/deliverables-studio'

const FRONTEND = path.resolve(__dirname, '..', '..', '..', '..')
const read = (rel: string) => readFileSync(path.join(FRONTEND, rel), 'utf8')

const intoView = vi.fn()
beforeEach(() => {
  width.compact = false
  nav.search = ''
  intoView.mockClear()
  Element.prototype.scrollIntoView = intoView
})
afterEach(cleanup)

describe('the tab strips keep their active tab visible', () => {
  it('Agents: on a compact viewport, from one shared hook', () => {
    width.compact = true
    nav.search = 'tab=skills'
    const { container } = render(<AgentManagementStudio />)
    expect(container.querySelector('.cc-tab.active')).toHaveTextContent('Skills')
    expect(intoView).toHaveBeenCalledWith({ block: 'nearest', inline: 'center' })
  })

  it('Deliverables: the same, and not a second copy of the behaviour', () => {
    width.compact = true
    nav.search = 'tab=templates'
    const { container } = render(<DeliverablesStudio />)
    expect(container.querySelector('.cc-tab.active')).toHaveTextContent('Templates')
    expect(intoView).toHaveBeenCalledWith({ block: 'nearest', inline: 'center' })
  })

  it('a desktop strip is left alone on both', () => {
    render(<AgentManagementStudio />)
    render(<DeliverablesStudio />)
    expect(intoView).not.toHaveBeenCalled()
  })

  it('every Studio tab strip in the app reaches the one hook', () => {
    // Tools' Studio frame has no tab strip — its controls are two `.cc-seg`
    // groups in the toolbar — so it is deliberately absent from this list.
    for (const f of [
      'components/command-center/command-center-shell.tsx',
      'components/assignments/studio/assignments-hub.tsx',
      'components/agents/studio/agent-management-studio.tsx',
      'components/deliverables/studio/deliverables-studio.tsx',
    ]) {
      expect(read(f), f).toContain("useTabStripScroll } from '@/hooks/use-tab-strip-scroll'")
      expect(read(f), f).toContain('ref={tabStrip}')
    }
    expect(read('components/tools/tools-dashboard.tsx')).not.toContain('cc-tabs')
  })
})

describe('the toolbars wrap instead of overflowing', () => {
  it('the search carries the class the compact rule reaches, on both pages that have one', () => {
    for (const f of [
      'components/agents/studio/agent-management-studio.tsx',
      'components/tools/tools-dashboard.tsx',
    ]) {
      expect(read(f), f).toContain('flex-1 cc-search')
    }
    const css = read('app/globals.css')
    const region = css.slice(css.indexOf('── Studio compact'))
    expect(region).toContain('.cc-toolbar .cc-search { flex-basis: 100%; }')
    expect(css, 'the toolbar already wraps on the desktop').toMatch(
      /\.cc-toolbar \{[^}]*flex-wrap: wrap/,
    )
  })

  it("Tools' Studio search is the one that changed — the classic branch is untouched", () => {
    const src = read('components/tools/tools-dashboard.tsx')
    const studio = src.indexOf("if (variant === 'studio')")
    expect(src.indexOf('flex-1 cc-search')).toBeGreaterThan(studio)
    expect(src.match(/cc-search/g) ?? []).toHaveLength(1)
  })
})

describe('the card grids are already one column on a phone', () => {
  it('the roster and the tools application grid start at one column', () => {
    // Both components are SHARED with Classic (M6), so the compact form must
    // come from their own base utility, not from an edit here.
    expect(read('components/agents/agent-roster.tsx')).toContain(
      'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
    )
    expect(read('components/tools/tools-dashboard.tsx')).toContain(
      'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
    )
  })

  it('the deliverables feed is swipeable card rows, not a multi-column grid', () => {
    for (const f of ['components/deliverables/type-row.tsx', 'components/deliverables/today-hero.tsx']) {
      expect(read(f), f).toContain('snap-x snap-mandatory')
      expect(read(f), f).not.toMatch(/grid-cols-\d/)
    }
  })
})
