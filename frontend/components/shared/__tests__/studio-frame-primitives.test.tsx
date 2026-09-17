/**
 * PRD-244 W5d (D8) — the shared frame primitives render the Studio frame in
 * the Studio style and their classic markup otherwise, so every page built
 * from them is a Studio page in both tones with Classic untouched.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import type { ReactNode } from 'react'
import { Bot } from 'lucide-react'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', setTheme: vi.fn() }) }))
vi.mock('@/hooks/use-system-config-api', () => ({ useSystemIcons: () => ({ data: {} }) }))
vi.mock('@/components/shared/premium-icon', () => ({ PremiumIcon: () => null }))
vi.mock('framer-motion', () => ({ motion: { div: ({ children, className }: { children: ReactNode; className?: string }) => <div className={className}>{children}</div> } }))

import { UiStyleProvider } from '@/contexts/ui-style-context'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar, statTone } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'

const Studio = ({ children }: { children: ReactNode }) => <UiStyleProvider initialStyle="studio">{children}</UiStyleProvider>

beforeEach(() => window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION))
afterEach(cleanup)

describe('PageHeader', () => {
  it('Studio: the editorial head — eyebrow, serif title, lede, actions slot', () => {
    const { container } = render(<Studio><PageHeader title="Knowledge" titleAccent="Bases" eyebrow="Workforce · what they know" lede="What they can read." actions={<button>Add</button>} /></Studio>)
    expect(container.querySelector('.cc-headrow .cc-head .cc-eyebrow')).toHaveTextContent('Workforce · what they know')
    expect(screen.getByTestId('page-title')).toHaveClass('cc-h1')
    expect(screen.getByTestId('page-title')).toHaveTextContent('Knowledge Bases')
    expect(container.querySelector('.cc-sub')).toHaveTextContent('What they can read.')
    expect(container.querySelector('.cc-actions button')).toHaveTextContent('Add')
    expect(container.querySelector('.gradient-text')).toBeNull()
  })
  it('Classic: the gradient title and subtitle, unchanged', () => {
    const { container } = render(<PageHeader title="Knowledge" titleAccent="Bases" subtitle="Docs" />)
    expect(container.querySelector('.gradient-text')).toHaveTextContent('Bases')
    expect(container.querySelector('.cc-head')).toBeNull()
  })
})

describe('StatsBar', () => {
  const stats = [
    { label: 'Total', value: 8, change: '8 agents', icon: Bot, iconColor: 'text-primary' },
    { label: 'Active', value: 7, change: '88% online', icon: Bot, iconColor: 'text-[hsl(var(--success))]' },
  ]
  it('Studio: the cc-stats strip with honest values and tones', () => {
    const { container } = render(<Studio><StatsBar stats={stats} /></Studio>)
    expect(container.querySelectorAll('.cc-stats .cell')).toHaveLength(2)
    expect(container.querySelector('.cell .v.ok')).toHaveTextContent('7')
    expect(screen.getByText('88% online')).toHaveClass('delta')
  })
  it('Studio: shows — while loading, never a placeholder number', () => {
    const { container } = render(<Studio><StatsBar stats={stats} loading /></Studio>)
    expect(container.querySelectorAll('.cell .v')[0]).toHaveTextContent('—')
  })
  it('Classic: the card grid, unchanged', () => {
    const { container } = render(<StatsBar stats={stats} />)
    expect(container.querySelector('.cc-stats')).toBeNull()
    expect(container.querySelectorAll('.glass-card')).toHaveLength(2)
  })
  it('maps the classic icon colour to the strip tone', () => {
    expect(statTone('text-[hsl(var(--success))]')).toBe('ok')
    expect(statTone('text-destructive')).toBe('err')
    expect(statTone('text-[hsl(var(--info))]')).toBe('info')
    expect(statTone('text-primary')).toBe('')
  })
})

describe('FilterTabs', () => {
  const tabs = [
    { value: 'a', label: 'Alpha', count: 3 },
    { value: 'b', label: 'Beta' },
  ]
  it('Studio: the cc-tabs strip drives the same Radix panels', () => {
    const onChange = vi.fn()
    const { container } = render(
      <Studio>
        <FilterTabs tabs={tabs} value="a" onValueChange={onChange} trailing={<span data-testid="trailing" />}>
          <TabsContent value="a">Panel A</TabsContent>
          <TabsContent value="b">Panel B</TabsContent>
        </FilterTabs>
      </Studio>,
    )
    expect(container.querySelectorAll('nav.cc-tabs button.cc-tab')).toHaveLength(2)
    expect(container.querySelector('.cc-tab.active')).toHaveTextContent('Alpha')
    expect(container.querySelector('.cc-tab-ct')).toHaveTextContent('3')
    expect(screen.getByText('Panel A')).toBeInTheDocument()
    expect(screen.queryByText('Panel B')).toBeNull()
    expect(screen.getByTestId('trailing')).toBeInTheDocument()
    expect(container.querySelector('[role="tablist"]')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Beta' }))
    expect(onChange).toHaveBeenCalledWith('b')
  })
  it('Classic: the Radix tab list, unchanged', () => {
    const { container } = render(
      <FilterTabs tabs={tabs} value="a" onValueChange={() => {}}>
        <TabsContent value="a">Panel A</TabsContent>
      </FilterTabs>,
    )
    expect(container.querySelector('[role="tablist"]')).not.toBeNull()
    expect(container.querySelector('nav.cc-tabs')).toBeNull()
  })
})
