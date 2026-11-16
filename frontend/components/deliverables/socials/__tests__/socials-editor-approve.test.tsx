/**
 * PRD-251D US-D101 (E3) — approve from the editor, against the editor's apiClient double.
 *
 * * A post that waits for approval shows the Queue's own controls to a person who may approve:
 *   "Approve · publishes HH:MM", Approve & post now, Request changes (back to Auto, as the
 *   Queue sends it) and Reject; a claim with no source asks the second confirmation.
 * * Approve sends the post's content hash (D6); while the editor holds unsaved edits it is off
 *   and the card says to save first.
 * * A viewer, and a post in any other status, see no approve controls.
 * * Once approved, the editor shows the post as approved, with its Activity card.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { toast } from 'sonner'

vi.mock('@/lib/api-client', async () => (await import('./socials-editor-harness')).apiModule)
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/deliverables/socials/socials-voice-picker', () => ({ SocialsVoicePicker: () => null }))
vi.mock('@/components/deliverables/socials/studio/socials-editor-activity', () => ({
  SocialsEditorActivity: () => <div data-testid="editor-activity" />,
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
}))

import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { APPROVE_CARD, SAVE_BEFORE_APPROVING, SEND_BACK_SUBMIT } from '@/components/deliverables/socials/studio/editor-approve-card'
import { FILE_OLDER_THAN_WORDS } from '@/components/deliverables/socials/studio/editor-submit'
import { POST_NOW_LABEL } from '@/components/deliverables/socials/socials-approve-post-now'
import { api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const APPROVE = 'Approve · publishes 12:00'
const card = () => screen.queryByRole('region', { name: APPROVE_CARD })

/** A post waiting for approval: on X, slotted for 12:00 UTC. */
const WAITING = post({
  status: 'needs_approval', content_hash: 'h-wait', planned_for: '2026-10-20T12:00:00Z', timezone: 'UTC',
  targets: [{ id: 't-x', toolkit: 'twitter', post_kind: 'image', options: {}, status: 'pending' } as any],
})

function renderEditor(p: SocialPost, role: Workspace['role'] = 'owner') {
  return renderWith(<SocialsEditor role={role} post={p} go={go} />)
}

beforeEach(() => { resetApi(); go.mockReset() })
afterEach(cleanup)

describe('approving in the editor', () => {
  it('a saved change to what the file shows turns Approve off and offers Submit, which makes the file again', async () => {
    // The post has its rendered file; the person then changes the brief and saves. The server keeps
    // the old file, so approving now would approve words the file does not show.
    const rendered = { ...WAITING, media: { '1:1': ['d-square'] } } as SocialPost
    renderEditor(rendered)
    const controls = await screen.findByRole('region', { name: APPROVE_CARD })
    expect(within(controls).getByRole('button', { name: APPROVE })).toBeEnabled()

    fireEvent.change(screen.getByRole('textbox', { name: 'Brief' }), { target: { value: 'A different point entirely' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))

    expect(await within(controls).findByText(FILE_OLDER_THAN_WORDS)).toBeInTheDocument()
    expect(within(controls).getByRole('button', { name: APPROVE })).toBeDisabled()
    expect(api.approveSocialPost).not.toHaveBeenCalled()
    // Submit is offered for a post that waits for approval, and it renders (the route allows it).
    fireEvent.click(screen.getByRole('button', { name: 'Submit for approval' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })

  it('a save that changes nothing the file shows, a post with no file yet, and a file of your own leave Approve as it was', async () => {
    const cases: Array<[SocialPost, string, string]> = [
      // The title is not on the file.
      [{ ...WAITING, media: { '1:1': ['d-square'] } } as SocialPost, 'Title', 'A better title'],
      // No file yet: there is nothing for the words to be newer than.
      [{ ...WAITING, media: {} } as SocialPost, 'Brief', 'A different point entirely'],
      // A video of the person's own shows none of the post's words, so no edit makes it older than them.
      [{ ...WAITING, format: 'video', media: { original: ['d-1'] } } as SocialPost, 'Brief', 'A different point entirely'],
    ]
    for (const [waiting, field, value] of cases) {
      vi.mocked(toast.success).mockClear()  // an earlier save's "Draft saved" must not answer for this one
      renderEditor(waiting)
      const controls = await screen.findByRole('region', { name: APPROVE_CARD })
      fireEvent.change(screen.getByRole('textbox', { name: field }), { target: { value } })
      fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))
      // The flag is set after the save has answered: wait for that, then one more turn of the loop.
      await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Draft saved'))
      await new Promise((resolve) => setTimeout(resolve, 0))
      expect(within(controls).queryByText(FILE_OLDER_THAN_WORDS)).toBeNull()
      expect(screen.queryByRole('button', { name: 'Submit for approval' })).toBeNull()
      cleanup()
    }
  })

  it("shows the Queue's controls for a post that waits for approval, to an owner and to an editor", async () => {
    for (const role of ['owner', 'editor'] as const) {
      renderEditor(WAITING, role)
      const controls = await screen.findByRole('region', { name: APPROVE_CARD })
      for (const name of [APPROVE, POST_NOW_LABEL, 'Request changes', 'Reject']) {
        expect(within(controls).getByRole('button', { name })).toBeEnabled()
      }
      expect(screen.queryByText(SAVE_BEFORE_APPROVING)).toBeNull()
      // The post already waits for approval: there is nothing to submit.
      expect(screen.queryByRole('button', { name: 'Submit for approval' })).toBeNull()
      cleanup()
    }
  })

  it("Approve calls the approve route with the post's content hash", async () => {
    renderEditor(WAITING)
    fireEvent.click(within(await screen.findByRole('region', { name: APPROVE_CARD })).getByRole('button', { name: APPROVE }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('post-1', 'h-wait'))
  })

  it('a claim with no source asks the second confirmation before anything is approved', async () => {
    renderEditor(post({ ...WAITING, variables: { members: { value: 412, claim: true } }, sources: {} }))
    const controls = await screen.findByRole('region', { name: APPROVE_CARD })
    fireEvent.click(within(controls).getByRole('button', { name: APPROVE }))
    expect(within(controls).getByRole('alertdialog', { name: 'Approve with unsourced claims' })).toHaveTextContent('members')
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })

  it('is off while the editor holds unsaved edits, and says to save first', async () => {
    renderEditor(WAITING)
    const controls = await screen.findByRole('region', { name: APPROVE_CARD })
    expect(within(controls).getByRole('button', { name: APPROVE })).toBeEnabled()

    fireEvent.change(screen.getByRole('textbox', { name: 'Copy' }), { target: { value: 'An edit not saved yet.' } })

    expect(within(controls).getByRole('button', { name: APPROVE })).toBeDisabled()
    expect(within(controls).getByRole('button', { name: POST_NOW_LABEL })).toBeDisabled()
    expect(within(controls).getByText(SAVE_BEFORE_APPROVING)).toBeInTheDocument()
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })

  it('Request changes sends the post back to Auto, as the Queue does', async () => {
    renderEditor(WAITING)
    const controls = await screen.findByRole('region', { name: APPROVE_CARD })
    fireEvent.click(within(controls).getByRole('button', { name: 'Request changes' }))
    fireEvent.change(within(controls).getByLabelText('What should change?'), { target: { value: 'Say Friday.' } })
    fireEvent.click(within(controls).getByRole('button', { name: SEND_BACK_SUBMIT }))
    await waitFor(() => expect(api.retakeSocialPost).toHaveBeenCalledWith('post-1', 'Say Friday.'))
    expect(api.requestSocialPostChanges).toHaveBeenCalledWith('post-1', 'Say Friday.')
  })

  it('a viewer sees no approve controls', async () => {
    renderEditor(WAITING, 'viewer')
    await screen.findByText('TikTok')
    expect(card()).toBeNull()
    expect(screen.queryByRole('button', { name: APPROVE })).toBeNull()
  })

  it('a post in any other status shows no approve controls', async () => {
    for (const status of ['draft', 'changes_requested', 'approved', 'scheduled'] as const) {
      renderEditor(post({ ...WAITING, status }))
      await screen.findByText('TikTok')
      expect(card()).toBeNull()
      expect(screen.queryByRole('button', { name: 'Request changes' })).toBeNull()
      cleanup()
    }
  })

  it('once approved, the editor shows the post as approved, with its Activity card', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const editor = (p: SocialPost) => (
      <QueryClientProvider client={client}><SocialsEditor role="owner" post={p} go={go} /></QueryClientProvider>
    )
    const view = render(editor(WAITING))
    fireEvent.click(within(await screen.findByRole('region', { name: APPROVE_CARD })).getByRole('button', { name: APPROVE }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('post-1', 'h-wait'))

    // The posts are read again, and the page hands the editor the post as the server has it now.
    view.rerender(editor(post({ ...WAITING, status: 'approved', approved_hash: 'h-wait', updated_at: '2026-10-14T09:00:00Z' })))

    expect(screen.getByTestId('socials-editor-status')).toHaveTextContent('Approved')
    expect(card()).toBeNull()
    expect(screen.getByTestId('editor-activity')).toBeInTheDocument()
  })
})
