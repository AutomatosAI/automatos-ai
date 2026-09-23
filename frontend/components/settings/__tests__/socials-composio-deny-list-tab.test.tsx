/**
 * PRD-251 S0.6 (D16) — the Composio deny list in Settings → System Settings.
 *
 * The `composio.denied_actions` row rides the same by-category plane as the
 * Socials switch: the tab shows the seeded slugs one per line, and saves the
 * JSON list the backend reads (upper-cased, de-duplicated). No row yet → it
 * says the migration is pending; an unreadable row says every action is
 * refused until it is saved again, and is shown verbatim for repair.
 *
 * P251-RVW-6: the backend refuses everything while the value is unreadable
 * and denies nothing once the list is empty, so this screen must never turn
 * the one into the other by accident. A line that is not an action slug
 * blocks Save, and an empty list needs a second, explicit confirmation.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'

import ComposioDenyListTab, {
  denyListEditorText,
  normalizeDeniedActions,
  notActionSlugs,
  parseDeniedActions,
} from '../ComposioDenyListTab'
import type { SystemSetting } from '@/lib/api/system-settings'

const D16 = [
  'HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE',
  'HIGGSFIELD_MCP_CANCEL_TRIAL_AUTO_RENEWAL',
  'HIGGSFIELD_MCP_CONFIRM_TRIAL_CANCEL',
  'HIGGSFIELD_MCP_CREATE_WEBSITE',
  'HIGGSFIELD_MCP_DEPLOY_WEBSITE',
  'HIGGSFIELD_MCP_PUBLISH_WEBSITE',
  'HIGGSFIELD_MCP_PARTICIPATE_IN_CONTEST',
  'HIGGSFIELD_MCP_APPS_INVOKE',
]

function seededRow(value: string | null): SystemSetting {
  return {
    id: 11,
    category: 'composio',
    key: 'denied_actions',
    value,
    value_type: 'json',
    description: null,
    is_sensitive: false,
    is_required: true,
    default_value: JSON.stringify(D16),
    validation_rules: null,
    created_at: '2026-09-23T00:00:00Z',
    updated_at: '2026-09-23T00:00:00Z',
    created_by: 'prd251',
  }
}

const textarea = () => screen.getByLabelText('Denied action slugs, one per line') as HTMLTextAreaElement
const saveButton = () => screen.getByRole('button', { name: /save deny list/i })

describe('ComposioDenyListTab (PRD-251 S0.6)', () => {
  it('says the migration is pending when the row is not seeded', () => {
    render(<ComposioDenyListTab settings={[]} onSave={vi.fn()} saving={false} onReset={vi.fn()} />)

    expect(screen.getByText(/has not been seeded yet/)).toBeInTheDocument()
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
  })

  it('shows the eight seeded D16 slugs, one per line', () => {
    render(
      <ComposioDenyListTab settings={[seededRow(JSON.stringify(D16))]} onSave={vi.fn()} saving={false} onReset={vi.fn()} />,
    )

    expect(textarea().value.split('\n')).toEqual(D16)
    expect(screen.getByText('8 actions')).toBeInTheDocument()
  })

  it('removing a slug saves the shorter JSON list the backend reads', () => {
    const onSave = vi.fn()
    render(
      <ComposioDenyListTab settings={[seededRow(JSON.stringify(D16))]} onSave={onSave} saving={false} onReset={vi.fn()} />,
    )

    const remaining = D16.filter((slug) => slug !== 'HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE')
    fireEvent.change(textarea(), { target: { value: remaining.join('\n') } })
    fireEvent.click(screen.getByRole('button', { name: /save deny list/i }))

    expect(onSave).toHaveBeenCalledWith({ denied_actions: JSON.stringify(remaining) })
    expect(screen.getByText('7 actions')).toBeInTheDocument()
  })

  it('upper-cases, trims and de-duplicates what the admin types', () => {
    const onSave = vi.fn()
    render(<ComposioDenyListTab settings={[seededRow('[]')]} onSave={onSave} saving={false} onReset={vi.fn()} />)

    fireEvent.change(textarea(), { target: { value: '  fal_ai_buy_credits \n\nFAL_AI_BUY_CREDITS\nkie_ai_topup' } })
    fireEvent.click(screen.getByRole('button', { name: /save deny list/i }))

    expect(onSave).toHaveBeenCalledWith({ denied_actions: JSON.stringify(['FAL_AI_BUY_CREDITS', 'KIE_AI_TOPUP']) })
  })

  it('warns that an unreadable stored list refuses every action', () => {
    render(<ComposioDenyListTab settings={[seededRow('not json')]} onSave={vi.fn()} saving={false} onReset={vi.fn()} />)

    expect(screen.getByRole('alert')).toHaveTextContent(/every Composio action is refused/)
    expect(textarea().value).toBe('not json')
  })

  it.each([
    'not json',
    '["HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE", 5]',
    '{"denied": ["HIGGSFIELD_MCP_DEPLOY_WEBSITE"]}',
    '[1]',
  ])('an unreadable stored value (%s) is shown verbatim, and an unedited Save writes nothing', (raw) => {
    const onSave = vi.fn()
    render(<ComposioDenyListTab settings={[seededRow(raw)]} onSave={onSave} saving={false} onReset={vi.fn()} />)

    expect(textarea().value).toBe(raw)
    expect(saveButton()).toBeDisabled()
    fireEvent.click(saveButton())

    expect(onSave).not.toHaveBeenCalled()
    expect(screen.getByText(/Not action slugs, so the list cannot be saved/)).toBeInTheDocument()
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
  })

  it('an unreadable stored value rewritten one slug per line saves as the JSON list', () => {
    const onSave = vi.fn()
    render(
      <ComposioDenyListTab
        settings={[seededRow('["HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE", 5]')]}
        onSave={onSave}
        saving={false}
        onReset={vi.fn()}
      />,
    )

    fireEvent.change(textarea(), {
      target: { value: 'HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE\nhiggsfield_mcp_deploy_website' },
    })
    fireEvent.click(saveButton())

    expect(onSave).toHaveBeenCalledWith({
      denied_actions: JSON.stringify(['HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE', 'HIGGSFIELD_MCP_DEPLOY_WEBSITE']),
    })
  })

  it('a JSON list pasted whole is not saved as one slug that matches nothing', () => {
    const onSave = vi.fn()
    render(
      <ComposioDenyListTab settings={[seededRow(JSON.stringify(D16))]} onSave={onSave} saving={false} onReset={vi.fn()} />,
    )

    fireEvent.change(textarea(), { target: { value: JSON.stringify(D16) } })

    expect(saveButton()).toBeDisabled()
    fireEvent.click(saveButton())
    expect(onSave).not.toHaveBeenCalled()
    expect(screen.getByText(/Not action slugs, so the list cannot be saved/)).toBeInTheDocument()
  })

  it.each([
    ['the seeded list', JSON.stringify(D16)],
    ['an unreadable value', 'not json'],
    ['a list already empty', '[]'],
  ])('saving an empty list, starting from %s, needs a second explicit confirmation', async (_start, raw) => {
    const onSave = vi.fn()
    render(<ComposioDenyListTab settings={[seededRow(raw)]} onSave={onSave} saving={false} onReset={vi.fn()} />)

    fireEvent.change(textarea(), { target: { value: '' } })
    fireEvent.click(saveButton())

    const confirmation = await screen.findByRole('alertdialog')
    expect(confirmation).toHaveTextContent(/every Composio action becomes runnable/)
    expect(confirmation).toHaveTextContent(/Higgsfield billing actions/)
    expect(onSave).not.toHaveBeenCalled()

    // Without the confirmation nothing is saved.
    fireEvent.click(within(confirmation).getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument())
    expect(onSave).not.toHaveBeenCalled()

    // With it, the empty list is saved.
    fireEvent.click(saveButton())
    fireEvent.click(within(await screen.findByRole('alertdialog')).getByRole('button', { name: 'Save the empty list' }))
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ denied_actions: '[]' }))
    expect(onSave).toHaveBeenCalledTimes(1)
  })

  it('resets through the category reset (back to the seeded list)', () => {
    const onReset = vi.fn()
    render(<ComposioDenyListTab settings={[seededRow('[]')]} onSave={vi.fn()} saving={false} onReset={onReset} />)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    expect(onReset).toHaveBeenCalledTimes(1)
  })

  it('parses and normalises like the backend', () => {
    expect(parseDeniedActions(null)).toEqual([])
    expect(parseDeniedActions('  ')).toEqual([])
    expect(parseDeniedActions('["A_B"]')).toEqual(['A_B'])
    expect(parseDeniedActions('{"a": 1}')).toBeNull()
    expect(parseDeniedActions('[1]')).toBeNull()
    expect(normalizeDeniedActions('a_b\n A_B \n\nc')).toEqual(['A_B', 'C'])
  })

  it('starts the editor from the stored list, or from an unreadable value verbatim', () => {
    expect(denyListEditorText(null)).toBe('')
    expect(denyListEditorText('["A_B","C"]')).toBe('A_B\nC')
    expect(denyListEditorText('{"a": 1}')).toBe('{"a": 1}')
    expect(notActionSlugs(['A_B', 'GMAIL_SEND_EMAIL2', '["A_B"]', 'NOT JSON', 'A,B'])).toEqual(['["A_B"]', 'NOT JSON', 'A,B'])
  })
})
