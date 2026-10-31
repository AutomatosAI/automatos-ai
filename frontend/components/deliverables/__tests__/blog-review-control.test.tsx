/**
 * PRD-264 EUAI-08b — the blog editor's "Mark as reviewed" control. An agent-written
 * post reads "AI-generated" until a workspace member marks it reviewed; a reviewed
 * post says when; a person's own post shows nothing. Unsaved edits must be saved
 * first (saving them would clear the review), and a viewer cannot review.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const state = vi.hoisted(() => ({ canEdit: true, mutate: vi.fn(), isLoading: false }))
vi.mock('@/hooks/use-blogs-api', () => ({
  useMarkPostReviewed: () => ({ mutate: state.mutate, isLoading: state.isLoading }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ canEdit: state.canEdit }),
}))

import { BlogReviewControl } from '@/components/deliverables/blog-review-control'
import { formFromPost, formMatchesPost, postDataFromForm } from '@/components/deliverables/blog-post-form'
import type { BlogPost } from '@/hooks/use-blogs-api'

const POST_ID = '3b9f1c2e-7a64-4d0e-9c51-2f8e6b1a0d08'

function post(overrides: Partial<BlogPost> = {}): BlogPost {
  return {
    id: POST_ID,
    workspace_id: '6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e08',
    author_agent_id: 12,
    author_name: 'Harbour Roasters',
    title: 'Harbour blend',
    slug: 'harbour-blend',
    excerpt: 'Roasted on Thursday.',
    content: 'Roasted on Thursday. Rested for a week.',
    cover_image_url: null,
    tags: ['coffee'],
    category: 'Roasting',
    status: 'draft',
    published_at: null,
    scheduled_for: null,
    seo_title: null,
    seo_description: null,
    reading_time_minutes: 1,
    view_count: 0,
    created_at: '2026-10-19T08:00:00+00:00',
    updated_at: '2026-10-19T08:00:00+00:00',
    ai_generated: true,
    reviewed_at: null,
    ...overrides,
  }
}

afterEach(() => {
  cleanup()
  state.canEdit = true
  state.isLoading = false
  state.mutate.mockReset()
})

describe('BlogReviewControl', () => {
  it('labels an unreviewed agent post AI-generated and marks it reviewed', () => {
    const { container } = render(<BlogReviewControl post={post()} hasUnsavedEdits={false} />)
    expect(screen.getByText('AI-generated')).toBeInTheDocument()
    expect(container.querySelector('[data-ai-generated="true"]')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /mark as reviewed/i }))
    expect(state.mutate).toHaveBeenCalledWith(POST_ID)
  })

  it('says when a reviewed post was reviewed, with no AI label', () => {
    render(<BlogReviewControl post={post({ ai_generated: false, reviewed_at: '2026-10-20T09:30:00+00:00' })} hasUnsavedEdits={false} />)
    expect(screen.getByTestId('blog-review-done')).toHaveTextContent(/Reviewed/)
    expect(screen.getByTestId('blog-review-done')).toHaveTextContent(/Editing the post clears the review/)
    expect(screen.queryByText('AI-generated')).toBeNull()
    expect(screen.queryByRole('button', { name: /mark as reviewed/i })).toBeNull()
  })

  it("shows nothing for a person's own post", () => {
    const { container } = render(<BlogReviewControl post={post({ author_agent_id: null, ai_generated: false })} hasUnsavedEdits={false} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('asks for unsaved edits to be saved first', () => {
    render(<BlogReviewControl post={post()} hasUnsavedEdits />)
    expect(screen.getByRole('button', { name: /mark as reviewed/i })).toBeDisabled()
    expect(screen.getByText(/Save your edits first/)).toBeInTheDocument()
  })

  it('a viewer cannot mark it reviewed', () => {
    state.canEdit = false
    render(<BlogReviewControl post={post()} hasUnsavedEdits={false} />)
    expect(screen.getByRole('button', { name: /mark as reviewed/i })).toBeDisabled()
  })
})

describe('the editor form', () => {
  it('matches the post it was filled from until a field changes', () => {
    const saved = post()
    const form = formFromPost(saved)
    expect(formMatchesPost(form, saved)).toBe(true)
    expect(formMatchesPost({ ...form, content: `${form.content} More.` }, saved)).toBe(false)
  })

  it('sends trimmed text and leaves empty optional fields out', () => {
    const form = { ...formFromPost(post()), title: '  Harbour blend  ', excerpt: '   ', tagsInput: 'coffee, , espresso ' }
    const data = postDataFromForm(form)
    expect(data.title).toBe('Harbour blend')
    expect(data.excerpt).toBeUndefined()
    expect(data.tags).toEqual(['coffee', 'espresso'])
  })
})
