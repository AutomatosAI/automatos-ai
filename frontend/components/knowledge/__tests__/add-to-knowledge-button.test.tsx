/**
 * PRE-11 (7 Oct): one Add to Knowledge button for a Deliverable, a ticket card and a
 * report. It takes its source's add and remove mutations and the owner's copy, if any.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/api-client', () => ({ apiClient: { request: vi.fn() } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import type { KnowledgeMutation } from '@/hooks/use-knowledge-copy'
import { AddToKnowledgeButton } from '../add-to-knowledge-button'

function mutation(isLoading = false) {
  return { mutate: vi.fn(), isLoading } as unknown as KnowledgeMutation
}

afterEach(() => cleanup())

describe('AddToKnowledgeButton', () => {
  it('offers Add to Knowledge when nothing was added, and adds on a click', () => {
    const add = mutation()
    const remove = mutation()
    render(<AddToKnowledgeButton documentId={null} add={add} remove={remove} keeps="the report" />)

    fireEvent.click(screen.getByRole('button', { name: /Add to Knowledge/ }))

    expect(add.mutate).toHaveBeenCalledTimes(1)
    expect(remove.mutate).not.toHaveBeenCalled()
    expect(screen.queryByText('Added to Knowledge')).toBeNull()
  })

  it('says it was added and removes the copy, naming what stays', () => {
    const add = mutation()
    const remove = mutation()
    render(<AddToKnowledgeButton documentId={42} add={add} remove={remove} keeps="the ticket" />)

    expect(screen.getByText('Added to Knowledge')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Add to Knowledge/ })).toBeNull()
    const removeButton = screen.getByRole('button', { name: 'Remove from Knowledge' })
    expect(removeButton).toHaveAttribute('title', 'Remove from Knowledge (the ticket stays)')

    fireEvent.click(removeButton)

    expect(remove.mutate).toHaveBeenCalledTimes(1)
    expect(add.mutate).not.toHaveBeenCalled()
  })

  it('cannot be clicked twice while a call is running', () => {
    render(<AddToKnowledgeButton documentId={undefined} add={mutation(true)} remove={mutation()} keeps="the report" />)
    expect(screen.getByRole('button', { name: /Add to Knowledge/ })).toBeDisabled()
  })
})
