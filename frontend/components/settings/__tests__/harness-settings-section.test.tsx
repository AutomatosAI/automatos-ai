/**
 * PRD-263 PR-1E-a — the HARNESS section, moved out of SystemLLMSettingsTab.tsx, shows
 * and sends what the tab did: an Enable switch, and once enabled a Run Schedule
 * (weekly, biweekly, monthly) and an Approval Mode (Full Auto, Manual).
 *
 * The <Select> is mocked as a native <select> (Radix's listbox needs pointer capture
 * and scrollIntoView, which jsdom does not implement), and the collapsible renders
 * its content so the controls are reachable without opening the card.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import React from 'react'

type MockSelectProps = { value: string; onValueChange: (value: string) => void; children: React.ReactNode }
type MockWrapperProps = { children: React.ReactNode }
type MockSelectItemProps = { value: string; children: React.ReactNode }

vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, children }: MockSelectProps) => (
    <select data-testid="harness-select" value={value} onChange={(e) => onValueChange(e.target.value)}>
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: MockWrapperProps) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: MockWrapperProps) => <>{children}</>,
  SelectItem: ({ value, children }: MockSelectItemProps) => <option value={value}>{children}</option>,
}))
vi.mock('@/components/ui/collapsible', () => ({
  Collapsible: ({ children }: MockWrapperProps) => <>{children}</>,
  CollapsibleTrigger: ({ children }: MockWrapperProps) => <>{children}</>,
  CollapsibleContent: ({ children }: MockWrapperProps) => <>{children}</>,
}))

import { HarnessSettingsSection } from '../harness-settings-section'
import { HarnessConfig, fallbackOrchestratorConfig } from '../orchestrator-settings-config'

const config = (harness: HarnessConfig) => ({ ...fallbackOrchestratorConfig(), harness })
const options = (select: HTMLElement) => within(select).getAllByRole('option').map((option) => option.textContent)

describe('HarnessSettingsSection (PRD-263 PR-1E-a, behaviour unchanged)', () => {
  it('shows the Enable switch and no schedule or mode while HARNESS is off', () => {
    render(
      <HarnessSettingsSection
        orchConfig={config({ enabled: false, schedule: 'weekly', mode: 'full_auto' })}
        onHarnessChange={vi.fn()}
      />,
    )

    expect(screen.getByText('Enable HARNESS')).toBeTruthy()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')
    expect(screen.queryAllByTestId('harness-select')).toHaveLength(0)
    expect(screen.queryByText('Active')).toBeNull()
  })

  it('sends enabled when the switch is turned on', () => {
    const onHarnessChange = vi.fn()
    render(
      <HarnessSettingsSection
        orchConfig={config({ enabled: false, schedule: 'weekly', mode: 'full_auto' })}
        onHarnessChange={onHarnessChange}
      />,
    )

    fireEvent.click(screen.getByRole('switch'))

    expect(onHarnessChange).toHaveBeenCalledWith('enabled', true)
  })

  it('offers the three schedules and the two approval modes once enabled', () => {
    render(
      <HarnessSettingsSection
        orchConfig={config({ enabled: true, schedule: 'biweekly', mode: 'full_auto' })}
        onHarnessChange={vi.fn()}
      />,
    )

    const [schedule, mode] = screen.getAllByTestId('harness-select') as HTMLSelectElement[]
    expect(options(schedule)).toEqual(['Weekly (Sunday 2 AM UTC)', 'Biweekly', 'Monthly'])
    expect(schedule.value).toBe('biweekly')
    expect(options(mode)).toEqual([
      'Full Auto — low-risk changes applied automatically',
      'Manual — all changes require your approval',
    ])
    expect(mode.value).toBe('full_auto')
    expect(screen.getByText('Active')).toBeTruthy()
    expect(screen.getByText('Changes with risk score 1-2 are applied automatically; risk 3+ queued for review')).toBeTruthy()
  })

  it('sends the schedule and mode picked, and says what Manual does', () => {
    const onHarnessChange = vi.fn()
    render(
      <HarnessSettingsSection
        orchConfig={config({ enabled: true, schedule: 'weekly', mode: 'manual' })}
        onHarnessChange={onHarnessChange}
      />,
    )

    const [schedule, mode] = screen.getAllByTestId('harness-select')
    fireEvent.change(schedule, { target: { value: 'monthly' } })
    fireEvent.change(mode, { target: { value: 'full_auto' } })

    expect(onHarnessChange).toHaveBeenNthCalledWith(1, 'schedule', 'monthly')
    expect(onHarnessChange).toHaveBeenNthCalledWith(2, 'mode', 'full_auto')
    expect(screen.getByText('Every proposed change will be queued as a board task for your review')).toBeTruthy()
  })

  it('shows no controls until the orchestrator config has loaded', () => {
    render(<HarnessSettingsSection orchConfig={null} onHarnessChange={vi.fn()} />)

    expect(screen.queryByRole('switch')).toBeNull()
    expect(screen.queryByText('Enable HARNESS')).toBeNull()
  })
})
