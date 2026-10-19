/**
 * PRD-263 US-017 — Settings offers the HARNESS modes that exist and says what each does.
 *
 * The tab offered "Full Auto — low-risk changes applied automatically" and "Manual",
 * behind an "Enable HARNESS" switch, with a Run Schedule of weekly (Sunday 2 AM UTC),
 * biweekly or monthly. HARNESS now reports and changes nothing (US-016): one mode
 * select, always shown, with Report only and Disabled in P1, no switch and no schedule.
 *
 * The <Select> is mocked as a native <select> (Radix's listbox needs pointer capture
 * and scrollIntoView, which jsdom does not implement), and the collapsible renders
 * its content so the select is reachable without opening the card.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import React from 'react'

type MockSelectProps = { value: string; onValueChange: (value: string) => void; children: React.ReactNode }
type MockWrapperProps = { children: React.ReactNode }
type MockSelectItemProps = { value: string; children: React.ReactNode }

vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, children }: MockSelectProps) => (
    <select data-testid="harness-mode-select" value={value} onChange={(e) => onValueChange(e.target.value)}>
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
import { HARNESS_HELP_TEXT, HarnessConfig, fallbackOrchestratorConfig } from '../orchestrator-settings-config'

const config = (harness: HarnessConfig) => ({ ...fallbackOrchestratorConfig(), harness })
const select = () => screen.getByTestId('harness-mode-select') as HTMLSelectElement

describe('HarnessSettingsSection (PRD-263 US-017)', () => {
  it('offers exactly the two modes P1 has, with no switch and no Full Auto', () => {
    render(
      <HarnessSettingsSection orchConfig={config({ schedule: 'weekly', mode: 'report_only' })} onHarnessChange={vi.fn()} />,
    )

    expect(within(select()).getAllByRole('option').map((option) => option.textContent)).toEqual([
      'Report only',
      'Disabled',
    ])
    expect(screen.queryByText(/full auto/i)).toBeNull()
    expect(screen.queryByRole('switch')).toBeNull()
    expect(screen.queryByText(/enable harness/i)).toBeNull()
  })

  it('shows an unknown stored value as Report only, with what Report only does', () => {
    render(<HarnessSettingsSection orchConfig={config({ mode: 'full_auto' })} onHarnessChange={vi.fn()} />)

    expect(select().value).toBe('report_only')
    expect(screen.getByText('HARNESS writes a weekly report on your agents and changes nothing.')).toBeTruthy()
  })

  it('sends the mode picked', () => {
    const onHarnessChange = vi.fn()
    render(
      <HarnessSettingsSection
        orchConfig={config({ schedule: 'weekly', mode: 'report_only' })}
        onHarnessChange={onHarnessChange}
      />,
    )

    fireEvent.change(select(), { target: { value: 'disabled' } })

    expect(onHarnessChange).toHaveBeenCalledWith('mode', 'disabled')
  })

  it('says when HARNESS runs and what it judges, and offers no schedule', () => {
    const { container } = render(
      <HarnessSettingsSection orchConfig={config({ schedule: 'weekly', mode: 'disabled' })} onHarnessChange={vi.fn()} />,
    )

    expect(screen.getByText(HARNESS_HELP_TEXT)).toBeTruthy()
    expect(screen.getByText('HARNESS does not run for this workspace.')).toBeTruthy()
    expect(screen.getAllByTestId('harness-mode-select')).toHaveLength(1)
    expect(container.textContent).not.toMatch(/sunday|biweekly|monthly|risk 3\+|applied automatically/i)
  })

  it('shows no select until the settings have loaded', () => {
    render(<HarnessSettingsSection orchConfig={null} onHarnessChange={vi.fn()} />)

    expect(screen.queryByTestId('harness-mode-select')).toBeNull()
  })
})
