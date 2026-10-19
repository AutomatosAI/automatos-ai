/**
 * PRD-260 P260-FX-05 (M41, review #47) — a sheet's clarification questions keep their hover.
 *
 * The question buttons carried `dark:bg-background/40 dark:border-gray-800/60` beside their
 * `hover:bg-warning/5 hover:border-warning/40`: dead until PRD-260's darkMode cherry-pick,
 * then winning in every dark tone (Tailwind emits the dark variant after hover: at the same
 * specificity), so the hover never showed. The block moved out of SheetArtifact (over the
 * component limit) into SheetClarifications, which ArtifactViewer renders above the sheet,
 * with token classes only.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
vi.mock('sonner', () => ({ toast }))
// The other artifact kinds are not under test; keep their dependencies out of the render.
vi.mock('../code-artifact', () => ({ CodeArtifact: () => null }))
vi.mock('../text-artifact', () => ({ TextArtifact: () => null }))
vi.mock('../document-artifact', () => ({ DocumentArtifact: () => null }))

import { SheetClarifications } from '../sheet-clarifications'
import { ArtifactViewer } from '../artifact-viewer'
import type { Artifact } from '@/types'

const QUESTIONS = ['Which financial year?', 'Only the EU region?']
const NEEDS_CLARIFICATION = { status: 'needs_clarification', clarifications: QUESTIONS, message: 'Two things first.' }
const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard')
const writeText = vi.fn(async (_text: string) => undefined)

beforeEach(() => {
  writeText.mockClear()
  toast.success.mockClear()
  toast.error.mockClear()
  Object.defineProperty(window.navigator, 'clipboard', { configurable: true, value: { writeText } })
})

afterEach(() => {
  cleanup()
  if (originalClipboard) Object.defineProperty(window.navigator, 'clipboard', originalClipboard)
  else delete (window.navigator as { clipboard?: unknown }).clipboard
})

describe('SheetClarifications (P260-FX-05)', () => {
  it('lists the questions, and a click copies one for the chat', async () => {
    render(<SheetClarifications metadata={NEEDS_CLARIFICATION} />)
    expect(screen.getByText('Clarification needed')).toBeInTheDocument()
    expect(screen.getByText('Two things first.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: QUESTIONS[1] }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(QUESTIONS[1]))
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Question copied (paste into chat to answer)'))
  })

  it('each question keeps its hover, with no bare dark: class over it', () => {
    render(<SheetClarifications metadata={NEEDS_CLARIFICATION} />)
    for (const question of QUESTIONS) {
      const option = screen.getByRole('button', { name: question })
      expect(option.className).toContain('hover:bg-warning/5')
      expect(option.className).toContain('hover:border-warning/40')
      expect(option.className).not.toMatch(/(^|\s)dark:(bg|border|text)-/)
    }
  })

  it('says so when the copy fails', async () => {
    writeText.mockRejectedValueOnce(new Error('denied'))
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    render(<SheetClarifications metadata={NEEDS_CLARIFICATION} />)
    fireEvent.click(screen.getByRole('button', { name: QUESTIONS[0] }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Failed to copy'))
    consoleError.mockRestore()
  })

  it('renders nothing unless the query asks for clarification', () => {
    const { container } = render(<SheetClarifications metadata={{ status: 'ok', clarifications: QUESTIONS }} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('ArtifactViewer shows the questions above the sheet', () => {
    const artifact: Artifact = { id: 'sheet-1', kind: 'sheet', title: 'Orders', content: '', metadata: NEEDS_CLARIFICATION }
    render(<ArtifactViewer artifact={artifact} onClose={vi.fn()} />)
    const question = screen.getByRole('button', { name: QUESTIONS[0] })
    const exportButton = screen.getByRole('button', { name: /Export CSV/ })
    expect(question.compareDocumentPosition(exportButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})
