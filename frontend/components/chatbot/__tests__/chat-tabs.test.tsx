/**
 * PRD-237 S3 — the tab strip: select, close, new, unread, last-tab rule.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { ChatTabs, type ChatTab } from '../chat-tabs'

const tabs: ChatTab[] = [
  { id: 'a', isDraft: false, title: 'Alpha', active: true, unread: false },
  { id: 'b', isDraft: false, title: 'Beta', active: false, unread: true },
  { id: null, isDraft: true, title: 'New chat', active: false, unread: false },
]

afterEach(cleanup)

describe('ChatTabs', () => {
  it('renders every tab, marks the active one and the unread one', () => {
    render(<ChatTabs tabs={tabs} onSelect={vi.fn()} onClose={vi.fn()} onNew={vi.fn()} />)
    const rendered = screen.getAllByRole('tab')
    expect(rendered).toHaveLength(3)
    expect(rendered[0]).toHaveAttribute('aria-selected', 'true')
    expect(rendered[1]).toHaveAttribute('aria-selected', 'false')
    expect(screen.getAllByLabelText('Unread')).toHaveLength(1)
  })

  it('selecting, closing and New chat call back with the tab', () => {
    const onSelect = vi.fn()
    const onClose = vi.fn()
    const onNew = vi.fn()
    render(<ChatTabs tabs={tabs} onSelect={onSelect} onClose={onClose} onNew={onNew} />)
    fireEvent.click(screen.getByTitle('Beta'))
    expect(onSelect).toHaveBeenCalledWith(tabs[1])
    fireEvent.click(screen.getByLabelText('Close Beta'))
    expect(onClose).toHaveBeenCalledWith(tabs[1])
    expect(onSelect).toHaveBeenCalledTimes(1) // close does not also select
    fireEvent.click(screen.getByLabelText('New chat'))
    expect(onNew).toHaveBeenCalledTimes(1)
  })

  it('the last remaining tab cannot be closed', () => {
    render(<ChatTabs tabs={[tabs[2]]} onSelect={vi.fn()} onClose={vi.fn()} onNew={vi.fn()} />)
    expect(screen.queryByLabelText(/^Close /)).toBeNull()
    expect(screen.getByRole('tab')).toHaveTextContent('New chat')
  })
})
