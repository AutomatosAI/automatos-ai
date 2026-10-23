/**
 * PRD-265 US-011 — the credit ledger on the Billing tab (owner/admin only).
 *
 * A 403 from GET /api/v1/billing/ledger hides the section and is not asked again; the rows
 * show date, what, model, signed credits and bucket; the pager walks the pages; the last 30
 * days' credits by model sit underneath; any other failure says so in one line.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { httpError, ledgerPage, renderWithClient } from './billing-fixtures'

const { requestMock } = vi.hoisted(() => ({ requestMock: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock } }))
vi.mock('@/lib/auth-edition', () => ({ isSaaS: true, isLocal: false }))

import { LedgerSection, LEDGER_FAILED } from '../ledger-section'

const LEDGER_TOTAL = 60

function pageOf(path: string): number {
  return Number(new URLSearchParams(path.split('?')[1] ?? '').get('page'))
}

beforeEach(() => {
  requestMock.mockReset()
})

afterEach(() => cleanup())

describe('Billing tab: the credit ledger (PRD-265 US-011)', () => {
  it('is hidden for a caller the API refuses (403), and is not asked twice', async () => {
    requestMock.mockRejectedValue(httpError(403, 'Workspace owner or admin required'))
    renderWithClient(<LedgerSection />)

    expect(screen.getByTestId('billing-ledger')).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByTestId('billing-ledger')).toBeNull())
    expect(requestMock).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(LEDGER_FAILED)).toBeNull()
  })

  it('shows each movement: date, what, model, signed credits and bucket; and the last 30 days by model', async () => {
    requestMock.mockImplementation(async (path: string) => ledgerPage(pageOf(path)))
    renderWithClient(<LedgerSection />)

    const table = await screen.findByTestId('billing-ledger-table')
    expect(requestMock).toHaveBeenCalledWith('/api/v1/billing/ledger?page=1&page_size=25')
    expect(within(table).getByText('AI usage')).toBeInTheDocument()
    expect(within(table).getByText('claude-haiku-5-5')).toBeInTheDocument()
    expect(within(table).getByText('-12')).toBeInTheDocument()
    expect(within(table).getByText('+500')).toBeInTheDocument()
    expect(within(table).getAllByText('Trial')).toHaveLength(2)
    const usage = screen.getByTestId('billing-usage-table')
    expect(within(usage).getByText('claude-haiku-5-5')).toBeInTheDocument()
    expect(within(usage).getByText('80')).toBeInTheDocument()
    expect(screen.getByText('1 credit = 1 US cent of AI usage.')).toBeInTheDocument()
    // Two rows fit on one page: no pager.
    expect(screen.queryByRole('button', { name: /Next/ })).toBeNull()
  })

  it('pages through the ledger', async () => {
    requestMock.mockImplementation(async (path: string) => ledgerPage(pageOf(path), LEDGER_TOTAL))
    renderWithClient(<LedgerSection />)

    expect(await screen.findByText('Page 1 of 3')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Previous/ })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: /Next/ }))

    expect(await screen.findByText('Page 2 of 3')).toBeInTheDocument()
    await waitFor(() => expect(requestMock).toHaveBeenCalledWith('/api/v1/billing/ledger?page=2&page_size=25'))
    await waitFor(() => expect(screen.getByRole('button', { name: /Previous/ })).toBeEnabled())

    fireEvent.click(screen.getByRole('button', { name: /Next/ }))
    expect(await screen.findByText('Page 3 of 3')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: /Next/ })).toBeDisabled())
  })

  it('says so in one line when the ledger fails to load for another reason', async () => {
    requestMock.mockRejectedValue(httpError(500, 'Internal Server Error'))
    renderWithClient(<LedgerSection />)

    expect(await screen.findByRole('alert')).toHaveTextContent(LEDGER_FAILED)
  })
})
