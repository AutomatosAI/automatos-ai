/**
 * PRD-238 S1 — the thinking channel folds when the answer takes over.
 */
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ReasoningBlock } from '../reasoning-block'

afterEach(cleanup)

const details = () => screen.getByTestId('reasoning-block') as HTMLDetailsElement

describe('ReasoningBlock', () => {
  it('is open and labelled Thinking while the model reasons', () => {
    render(<ReasoningBlock text="Let me check the fleet" streaming answerStarted={false} />)
    expect(details().open).toBe(true)
    expect(screen.getByText('Thinking…')).toBeInTheDocument()
    expect(screen.getByText('Let me check the fleet')).toBeInTheDocument()
  })

  it('folds the moment the answer starts, and reads as a thought process afterwards', () => {
    const { rerender } = render(<ReasoningBlock text="hmm" streaming answerStarted={false} />)
    expect(details().open).toBe(true)
    rerender(<ReasoningBlock text="hmm" streaming answerStarted />)
    expect(details().open).toBe(false)
    expect(screen.getByText('Thought process')).toBeInTheDocument()
    rerender(<ReasoningBlock text="hmm" streaming={false} answerStarted />)
    expect(details().open).toBe(false)
  })

  it('renders folded on reload and nothing at all without text', () => {
    render(<ReasoningBlock text="stored" streaming={false} answerStarted />)
    expect(details().open).toBe(false)
    cleanup()
    const { container } = render(<ReasoningBlock text="" streaming={false} answerStarted={false} />)
    expect(container).toBeEmptyDOMElement()
  })
})
