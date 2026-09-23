/**
 * PRD-251 S0.1 — the Socials master switch in Settings → System Settings.
 *
 * The `socials.enabled` row rides the same by-category plane as Voice: the
 * tab shows the seeded state, flips it, and saves the string the backend reads
 * ('true' / 'false'). No row yet → it says the migration is pending.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

import SocialsSettingsTab from '../SocialsSettingsTab'
import type { SystemSetting } from '@/lib/api/system-settings'

function seededRow(value: string | null): SystemSetting {
  return {
    id: 7,
    category: 'socials',
    key: 'enabled',
    value,
    value_type: 'boolean',
    description: null,
    is_sensitive: false,
    is_required: true,
    default_value: 'false',
    validation_rules: null,
    created_at: '2026-09-23T00:00:00Z',
    updated_at: '2026-09-23T00:00:00Z',
    created_by: 'prd251',
  }
}

describe('SocialsSettingsTab (PRD-251 S0.1)', () => {
  it('says the migration is pending when the row is not seeded', () => {
    render(<SocialsSettingsTab settings={[]} onSave={vi.fn()} saving={false} onReset={vi.fn()} />)

    expect(screen.getByText(/have not been seeded yet/)).toBeInTheDocument()
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
  })

  it('shows OFF from the seeded row, flips it and saves the string the backend reads', () => {
    const onSave = vi.fn()
    render(
      <SocialsSettingsTab settings={[seededRow('false')]} onSave={onSave} saving={false} onReset={vi.fn()} />,
    )

    expect(screen.getByText('OFF')).toBeInTheDocument()
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'false')

    fireEvent.click(screen.getByRole('switch'))
    expect(screen.getByText('ON')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /save socials settings/i }))
    expect(onSave).toHaveBeenCalledWith({ enabled: 'true' })
  })

  it('reads ON from the stored value', () => {
    render(
      <SocialsSettingsTab settings={[seededRow('true')]} onSave={vi.fn()} saving={false} onReset={vi.fn()} />,
    )

    expect(screen.getByText('ON')).toBeInTheDocument()
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
  })

  it('resets through the category reset', () => {
    const onReset = vi.fn()
    render(
      <SocialsSettingsTab settings={[seededRow('true')]} onSave={vi.fn()} saving={false} onReset={onReset} />,
    )

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    expect(onReset).toHaveBeenCalledTimes(1)
  })
})
