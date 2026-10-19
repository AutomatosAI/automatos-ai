/**
 * PRD-264 EUAI-08a — the composer's "Realistic AI people, places, products or voice: label
 * as AI-generated", and the receipt that shows it, against a mocked apiClient.
 *
 * - The switch shows the backend's setting (GET /ai-labels, `deepfake_label`): on by
 *   default while the post carries AI media, with the reason a person switching it off
 *   is told ("you are responsible for disclosing deep fakes").
 * - Switching it calls PUT /api/socials/posts/{id}/ai-label with the new value.
 * - A role that cannot author posts sees it, and cannot switch it; a new post is told to
 *   save first.
 * - The receipt states the setting in the backend's words.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

const api = vi.hoisted(() => ({ get: vi.fn(), put: vi.fn(), retrySocialPost: vi.fn() }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))

import {
  AI_LABEL_DEFAULT, AI_LABEL_NEW_POST, AI_LABEL_SETTING, EditorAiLabelCard,
} from '@/components/deliverables/socials/studio/editor-ai-label-card'
import { SocialsPostReceipts } from '@/components/deliverables/socials/socials-post-receipts'
import type { SocialPost } from '@/lib/api-client'

const HASH = 'a'.repeat(64)
const REASON = 'You are responsible for disclosing deep fakes.'
const ON_LINE = `${AI_LABEL_SETTING}: on. The caption starts with "AI-generated."`
const AI_MEDIA = { carries: true, footage: [], stills: [], voice: 'kokoro', own: [], recorded: true }

function setting(over: Record<string, unknown> = {}) {
  return {
    name: AI_LABEL_SETTING, setting: null, default: true, on: true, label: 'AI-generated.', reason: REASON,
    video_labelled: true, line: ON_LINE, ...over,
  }
}

function post(overrides: Partial<SocialPost> = {}): SocialPost {
  return {
    id: 'p1', workspace_id: 'w1', created_by: 'u1', title: 'Countdown', brief: null, copy: { base: 'Three weeks to go.' },
    format: 'video', template_id: 't1', variables: {}, sources: {}, media: {}, status: 'draft', content_hash: HASH,
    approved_hash: null, approved_by: null, approved_at: null, override_unsourced: false, review_log: [],
    scheduled_for: null, timezone: null, planned_for: null, length_seconds: 30, targets: [],
    created_at: '2026-10-20T09:00:00Z', updated_at: '2026-10-20T09:00:00Z', ...overrides,
  }
}

function renderWith(element: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{element}</QueryClientProvider>)
}

beforeEach(() => {
  api.get.mockReset().mockResolvedValue({ ai_media: AI_MEDIA, deepfake_label: setting(), targets: [] })
  api.put.mockReset().mockResolvedValue(post())
})
afterEach(() => cleanup())

describe('the AI-generated label in the composer', () => {
  it('is on by default for a post with AI media, with the reason switching it off is the customer\'s call', async () => {
    renderWith(<EditorAiLabelCard post={post()} role="owner" />)
    const toggle = await screen.findByRole('switch', { name: AI_LABEL_SETTING })
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText(AI_LABEL_DEFAULT)).toBeInTheDocument()
    expect(screen.getByTestId('socials-ai-label-reason')).toHaveTextContent(
      'Switching it off: you are responsible for disclosing deep fakes.',
    )
    expect(api.get).toHaveBeenCalledWith('/api/socials/posts/p1/ai-labels')  // before any channel is ticked
  })

  it('switches it off for this post through PUT /ai-label', async () => {
    renderWith(<EditorAiLabelCard post={post()} role="owner" />)
    fireEvent.click(await screen.findByRole('switch', { name: AI_LABEL_SETTING }))
    await waitFor(() => expect(api.put).toHaveBeenCalledWith('/api/socials/posts/p1/ai-label', { on: false }))
  })

  it('shows a switched-off post as off, with the reason', async () => {
    api.get.mockResolvedValue({
      ai_media: AI_MEDIA, targets: [],
      deepfake_label: setting({ setting: false, on: false, line: `${AI_LABEL_SETTING}: off for this post. ${REASON}` }),
    })
    renderWith(<EditorAiLabelCard post={post()} role="owner" />)
    expect(await screen.findByRole('switch', { name: AI_LABEL_SETTING })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByTestId('socials-ai-label-reason')).toHaveTextContent(REASON)
    fireEvent.click(screen.getByRole('switch', { name: AI_LABEL_SETTING }))
    await waitFor(() => expect(api.put).toHaveBeenCalledWith('/api/socials/posts/p1/ai-label', { on: true }))
  })

  it('cannot be switched by a role that cannot author posts', async () => {
    renderWith(<EditorAiLabelCard post={post()} role="viewer" />)
    expect(await screen.findByRole('switch', { name: AI_LABEL_SETTING })).toBeDisabled()
  })

  it('asks a new post to be saved first', () => {
    renderWith(<EditorAiLabelCard post={null} role="owner" />)
    expect(screen.getByText(AI_LABEL_NEW_POST)).toBeInTheDocument()
    expect(api.get).not.toHaveBeenCalled()
  })
})

describe('the receipt', () => {
  it('shows the setting in the backend\'s words', async () => {
    const target = {
      id: 't-linkedin', toolkit: 'linkedin', post_kind: 'text', options: {}, status: 'pending' as const, attempts: 0,
      remote_id: null, permalink: null, error: null, published_at: null, notes: [],
    }
    const added = 'This post carries a synthetic voice (Kokoro), and LinkedIn takes no AI label through its API, ' +
      'so the caption carries the disclosure: "AI-generated." was added at its start.'
    api.get.mockResolvedValue({
      ai_media: AI_MEDIA, deepfake_label: setting(),
      targets: [{ id: 't-linkedin', toolkit: 'linkedin', post_kind: 'text', ai_label: added, at_publish: false }],
    })
    renderWith(<SocialsPostReceipts post={post({ status: 'approved', targets: [target] })} role="owner" />)
    expect(await screen.findByTestId('socials-ai-label-setting')).toHaveTextContent(ON_LINE)
    expect(screen.getByTestId('socials-ai-label-linkedin-text')).toHaveTextContent(added)
  })
})
