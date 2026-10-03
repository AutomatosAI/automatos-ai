/**
 * F243 — a redo that failed keeps the draft it was correcting on the card's face.
 * Night 7: #0171 and #0174 ran out of credit on their redo, and the good email
 * was only in the card's history.
 */
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import type { BoardTask } from '@/types/board'

import { KeptDraft } from '../board-task-viewer'

function card(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '171', type: 'task', name: "Rosa's apology", status: 'failed', priority: 'medium', tags: [],
    review_mode: 'auto', source_id: '171', error_message: "The AI provider's account ran out of credit", ...over }
}

afterEach(cleanup)

describe('KeptDraft', () => {
  it('shows the draft a failed redo kept, under the failure', () => {
    render(<KeptDraft task={card({ kept_draft: 'Hi Rosa, ... Gerard' })} />)
    expect(screen.getByText('Hi Rosa, ... Gerard')).toBeInTheDocument()
    expect(screen.getByText(/this run failed; the draft is kept/)).toBeInTheDocument()
  })

  it('shows nothing when the run has a result of its own, or there is no draft', () => {
    const { container } = render(<KeptDraft task={card({ kept_draft: 'Hi Rosa', result: 'A new draft' })} />)
    expect(container).toBeEmptyDOMElement()
    cleanup()
    const { container: none } = render(<KeptDraft task={card()} />)
    expect(none).toBeEmptyDOMElement()
  })
})
