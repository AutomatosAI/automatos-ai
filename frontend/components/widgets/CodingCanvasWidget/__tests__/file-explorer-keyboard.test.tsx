/**
 * Issue #1090 — a file tree row is a real button: Enter activates it like a click.
 * fireEvent.keyDown would only prove a hand-written handler; userEvent reproduces
 * the browser's own activation, which is what a native button relies on.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import React from 'react'
import type { WorkspaceFileEntry } from '../../types'
import { FileExplorer } from '../FileExplorer'

const ENTRIES: WorkspaceFileEntry[] = [
  { name: 'notes.md', path: 'notes.md', type: 'file', size: 12 },
]

afterEach(() => cleanup())

describe('FileExplorer keyboard', () => {
  it('opens a file with Enter just as a click does', async () => {
    const user = userEvent.setup()
    const onFileSelect = vi.fn()
    render(<FileExplorer entries={ENTRIES} onFileSelect={onFileSelect} onDirectoryToggle={vi.fn()} />)
    const row = screen.getByRole('button', { name: 'notes.md' })
    expect(row.tagName).toBe('BUTTON')

    fireEvent.click(row)
    expect(onFileSelect).toHaveBeenCalledWith('notes.md')

    onFileSelect.mockClear()
    row.focus()
    await user.keyboard('{Enter}')
    expect(onFileSelect).toHaveBeenCalledWith('notes.md')
  })
})
