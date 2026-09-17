/**
 * PRD-246 US-006 — the compact form of the three shared frame primitives.
 *
 * `PageHeader`, `StatsBar` and `FilterTabs` converted 24/11/9 pages to Studio
 * in PRD-244 W5d–g, so a change here reaches Team, Analytics, Settings and
 * its sub-pages, Knowledge Base, Marketplace and its sub-pages, mission
 * detail and the three admin surfaces at once. That leverage cuts both ways,
 * so this asserts BOTH branches: the Studio structure the compact rules act
 * on, and the Classic markup, which must stay exactly as it renders today.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { readFileSync } from 'fs'
import type { ReactNode } from 'react'
import path from 'path'
import { Bot } from 'lucide-react'

const width = vi.hoisted(() => ({ compact: false }))

vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => width.compact,
  useIsTabletOrBelow: () => width.compact,
}))
vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', setTheme: vi.fn() }) }))
vi.mock('@/hooks/use-system-config-api', () => ({ useSystemIcons: () => ({ data: {} }) }))
vi.mock('@/components/shared/premium-icon', () => ({ PremiumIcon: () => null }))
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, className }: { children: ReactNode; className?: string }) => (
      <div className={className}>{children}</div>
    ),
  },
}))

import { UiStyleProvider } from '@/contexts/ui-style-context'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'

const Studio = ({ children }: { children: ReactNode }) => (
  <UiStyleProvider initialStyle="studio">{children}</UiStyleProvider>
)

const FRONTEND = path.resolve(__dirname, '..', '..', '..')
const css = readFileSync(path.join(FRONTEND, 'app', 'globals.css'), 'utf8')
const region = css.slice(css.indexOf('── Studio compact'), css.indexOf('end of the compact region'))
const phone = region.slice(region.indexOf('@media (max-width: 767px)'))

const stats = [
  { label: 'Total', value: 8, change: '8 agents', icon: Bot, iconColor: 'text-primary' },
  { label: 'Active', value: 7, change: '88% online', icon: Bot, iconColor: 'text-[hsl(var(--success))]' },
]
const tabs = [
  { value: 'a', label: 'Alpha', count: 3 },
  { value: 'b', label: 'Beta' },
]

const intoView = vi.fn()
beforeEach(() => {
  width.compact = false
  intoView.mockClear()
  Element.prototype.scrollIntoView = intoView
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
})
afterEach(cleanup)

describe('Studio: the compact form', () => {
  it('PageHeader — the head and its actions are siblings the head-row stacks', () => {
    width.compact = true
    const { container } = render(
      <Studio>
        <PageHeader title="Knowledge" titleAccent="Bases" eyebrow="Workforce" lede="What they read." actions={<button>Add</button>} />
      </Studio>,
    )
    const row = container.querySelector('.cc-headrow')!
    expect(row.children).toHaveLength(2)
    expect(row.children[0]).toHaveClass('cc-head')
    expect(row.children[1]).toHaveClass('cc-actions')
    // …and the rule that stacks them, in the one compact region.
    expect(phone).toContain('.cc-headrow { flex-wrap: wrap; }')
    expect(phone).toContain('.cc-headrow > .cc-head { flex-basis: 100%; }')
    expect(phone).toContain('.cc-actions { flex-wrap: wrap; }')
  })

  it('StatsBar — the strip is `.cc-stats`, which the region already takes 2-up', () => {
    width.compact = true
    const { container } = render(<Studio><StatsBar stats={stats} /></Studio>)
    expect(container.querySelector('.cc-stats')).not.toBeNull()
    expect(container.querySelectorAll('.cc-stats .cell')).toHaveLength(2)
    // One rule for the family, not a second one for this primitive (US-002).
    expect(phone).toContain(
      ':is(.studio, .cc-page) .cc-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }',
    )
  })

  it('FilterTabs — the strip scrolls its active tab into view on a phone', () => {
    width.compact = true
    render(
      <Studio>
        <FilterTabs tabs={tabs} value="b" onValueChange={vi.fn()}>
          <TabsContent value="b">Panel B</TabsContent>
        </FilterTabs>
      </Studio>,
    )
    expect(intoView).toHaveBeenCalledWith({ block: 'nearest', inline: 'center' })
  })

  it('FilterTabs — a desktop strip is left alone', () => {
    render(
      <Studio>
        <FilterTabs tabs={tabs} value="b" onValueChange={vi.fn()}>
          <TabsContent value="b">Panel B</TabsContent>
        </FilterTabs>
      </Studio>,
    )
    expect(intoView).not.toHaveBeenCalled()
  })
})

describe('Classic renders exactly as it did — at every width', () => {
  it('PageHeader keeps the gradient accent and no Studio head', () => {
    width.compact = true
    const { container } = render(<PageHeader title="Knowledge" titleAccent="Bases" subtitle="Docs" />)
    expect(container.querySelector('.gradient-text')).toHaveTextContent('Bases')
    expect(container.querySelector('.cc-head')).toBeNull()
    expect(container.querySelector('.cc-headrow')).toBeNull()
  })

  it('StatsBar keeps the card grid and no strip', () => {
    width.compact = true
    const { container } = render(<StatsBar stats={stats} />)
    expect(container.querySelector('.cc-stats')).toBeNull()
    expect(container.querySelectorAll('.glass-card')).toHaveLength(2)
  })

  it('FilterTabs keeps the Radix tablist and no cc-tabs strip', () => {
    width.compact = true
    const { container } = render(
      <FilterTabs tabs={tabs} value="a" onValueChange={vi.fn()}>
        <TabsContent value="a">Panel A</TabsContent>
      </FilterTabs>,
    )
    expect(container.querySelector('[role="tablist"]')).not.toBeNull()
    expect(container.querySelector('nav.cc-tabs')).toBeNull()
    expect(screen.getByText('Panel A')).toBeInTheDocument()
    // The ref is never attached in this branch, so the hook cannot touch it.
    expect(intoView).not.toHaveBeenCalled()
  })

  it('no compact rule in the region can reach a Classic page', () => {
    // Every rule is scoped to `.studio` or to a Studio-only root; a Classic
    // page carries neither class.
    const declarations = region.replace(/\/\*[\s\S]*?\*\//g, '')
    const unscoped = declarations
      .split('\n')
      .filter((l) => /^\s+\.(cc|sh|entry|mis|pb|mkt|status|pg)-/.test(l))
      .filter((l) => !/^\s+\.(cc-page|cc-cal-root|sh-chat|sh-shell)\b/.test(l))
    expect(unscoped).toEqual([])
  })
})
