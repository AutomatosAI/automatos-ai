/**
 * PRD-251D US-D101 (E3) — approve from the editor, against the editor's apiClient double.
 *
 * * A post that waits for approval shows the Queue's own controls to a person who may approve:
 *   the one Approve, labelled for what it does ("Approve · posts HH:MM" for a slot ahead, "Approve
 *   & post now to X" for no slot, which one click approves and the server publishes), Request
 *   changes (back to Auto, as the Queue sends it) and Reject; there is no separate Approve & post
 *   now; a claim with no source asks the second confirmation.
 * * Approve sends the post's content hash (D6); while the editor holds unsaved edits it is off
 *   and the card says to save first.
 * * A viewer, and a post in any other status, see no approve controls.
 * * Once approved, the editor shows the post as approved, with its Activity card. Approved with no
 *   time (unscheduled, say), the line where the Approve card was says nothing will post, and that
 *   Schedule and Publish now are in Activity further down (the last card on the page).
 * * US-D108: a post whose file is older than its words (its own `file_current`, which the server
 *   keeps) shows the Queue's sentence in place of a working Approve, with Render again, and is
 *   offered Submit, which makes the file again. The editor holds nothing for the sitting: Approve
 *   follows the post as the server answers it.
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
// The Activity card has its own tests; here it shows whether the editor asks it to leave out the
// no-time line the editor already shows at its top.
vi.mock('@/components/deliverables/socials/studio/socials-editor-activity', () => ({
  SocialsEditorActivity: ({ hideNoTime }: { hideNoTime?: boolean }) => (
    <div data-testid="editor-activity" data-hide-no-time={String(Boolean(hideNoTime))} />
  ),
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
}))

import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { APPROVE_CARD, SAVE_BEFORE_APPROVING, SEND_BACK_SUBMIT } from '@/components/deliverables/socials/studio/editor-approve-card'
import { RENDER_AGAIN } from '@/components/deliverables/socials/socials-post-review'
import { FILE_OLDER_THAN_WORDS } from '@/components/deliverables/socials/socials-review'
import { APPROVE_AND_POST_NOW, APPROVE_POSTS_AT } from '@/components/deliverables/socials/studio/queue-model'
import { APPROVED_NO_TIME_NOTE, CONTROLS_IN_ACTIVITY } from '@/components/deliverables/socials/studio/post-notices'
import { api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
/** Approve on WAITING, whose 12:00 slot is still ahead at the faked clock. */
const APPROVE = `${APPROVE_POSTS_AT} 12:00`
const NOW = new Date('2026-10-14T10:00:00Z')
const card = () => screen.queryByRole('region', { name: APPROVE_CARD })

/** A post waiting for approval: on X, slotted for 12:00 UTC. */
const WAITING = post({
  status: 'needs_approval', content_hash: 'h-wait', planned_for: '2026-10-20T12:00:00Z', timezone: 'UTC',
  targets: [{ id: 't-x', toolkit: 'twitter', post_kind: 'image', options: {}, status: 'pending' } as any],
})
/** The same post with no slot: its Approve publishes it at once. */
const UNTIMED = post({ ...WAITING, planned_for: null })

function renderEditor(p: SocialPost, role: Workspace['role'] = 'owner') {
  return renderWith(<SocialsEditor role={role} post={p} go={go} />)
}

beforeEach(() => {
  // Approve's label reads the slot against the clock: 10:00 on 14 Oct, so the 20 Oct slot is ahead.
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(NOW)
  resetApi()
  go.mockReset()
})
afterEach(() => { cleanup(); vi.useRealTimers() })

describe('approving in the editor', () => {
  /** A post waiting for approval with its rendered file. */
  const RENDERED = { ...WAITING, media: { '1:1': ['d-square'] } } as SocialPost

  it('a post whose file is older than its words shows the sentence in place of a working Approve, and Render again renders it', async () => {
    // The server says the file was made before the post's last change to what a render reads.
    renderEditor({ ...RENDERED, file_current: false })
    const controls = await screen.findByRole('region', { name: APPROVE_CARD })
    expect(within(controls).getByText(FILE_OLDER_THAN_WORDS)).toBeInTheDocument()
    expect(within(controls).getByRole('button', { name: APPROVE })).toBeDisabled()

    fireEvent.click(within(controls).getByRole('button', { name: RENDER_AGAIN }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1'))
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })

  it('offers Submit for such a post, and Submit makes the file again (a render is allowed while it waits)', async () => {
    renderEditor({ ...RENDERED, file_current: false })
    await screen.findByRole('region', { name: APPROVE_CARD })
    fireEvent.click(screen.getByRole('button', { name: 'Submit for approval' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })

  it("Approve follows the post's own flag as the server answers it: the editor holds nothing for the sitting", async () => {
    vi.mocked(toast.success).mockClear()  // an earlier test's "Draft saved" must not answer for this one
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const editor = (p: SocialPost) => (
      <QueryClientProvider client={client}><SocialsEditor role="owner" post={p} go={go} /></QueryClientProvider>
    )
    const view = render(editor(RENDERED))
    await screen.findByRole('region', { name: APPROVE_CARD })

    // The brief is not on the file: saved, the server answers the post with its file still current.
    fireEvent.change(screen.getByRole('textbox', { name: 'Brief' }), { target: { value: 'A different point entirely' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Draft saved'))
    view.rerender(editor({ ...RENDERED, brief: 'A different point entirely', file_current: true, updated_at: '2026-10-14T09:00:00Z' }))
    let controls = screen.getByRole('region', { name: APPROVE_CARD })
    expect(within(controls).getByRole('button', { name: APPROVE })).toBeEnabled()
    expect(within(controls).queryByText(FILE_OLDER_THAN_WORDS)).toBeNull()
    expect(screen.queryByRole('button', { name: 'Submit for approval' })).toBeNull()

    // A field is: the server answers with `file_current` false, and Approve waits for the file.
    view.rerender(editor({ ...RENDERED, file_current: false, updated_at: '2026-10-14T09:05:00Z' }))
    controls = screen.getByRole('region', { name: APPROVE_CARD })
    expect(within(controls).getByText(FILE_OLDER_THAN_WORDS)).toBeInTheDocument()
    expect(within(controls).getByRole('button', { name: APPROVE })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Submit for approval' })).toBeInTheDocument()
  })

  it("shows the Queue's controls for a post that waits for approval, to an owner and to an editor", async () => {
    for (const role of ['owner', 'editor'] as const) {
      renderEditor(WAITING, role)
      const controls = await screen.findByRole('region', { name: APPROVE_CARD })
      for (const name of [APPROVE, 'Request changes', 'Reject']) {
        expect(within(controls).getByRole('button', { name })).toBeEnabled()
      }
      expect(within(controls).queryByRole('button', { name: 'Approve & post now' })).toBeNull()
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

  it('with no slot, Approve says it posts now to its channels, and one click approves it: the server publishes it', async () => {
    renderEditor(UNTIMED)
    const controls = await screen.findByRole('region', { name: APPROVE_CARD })
    fireEvent.click(within(controls).getByRole('button', { name: `${APPROVE_AND_POST_NOW} X` }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('post-1', 'h-wait', { postNow: true }))
    expect(api.approveSocialPost).toHaveBeenCalledTimes(1)
    // Nothing more is asked, and the screen never publishes it itself: the approve route does.
    expect(within(controls).queryByRole('alertdialog')).toBeNull()
    expect(api.publishSocialPostNow).not.toHaveBeenCalled()
    expect(screen.queryByText(APPROVED_NO_TIME_NOTE, { exact: false })).toBeNull()
  })

  it('approved with no time, the line where the Approve card was says nothing will post, and where Schedule and Publish now are', async () => {
    // Approved with neither slot (unscheduled, say): nothing posts it until a person gives it a time.
    renderEditor(post({ ...UNTIMED, status: 'approved', approved_hash: 'h-wait', updated_at: '2026-10-14T09:00:00Z' }))
    await screen.findByText('TikTok')
    const notice = screen.getByText(`${APPROVED_NO_TIME_NOTE} ${CONTROLS_IN_ACTIVITY}`)
    // Said once: the Activity card is asked to leave its own copy of the line out.
    expect(screen.getAllByText(APPROVED_NO_TIME_NOTE, { exact: false })).toHaveLength(1)
    expect(screen.getByTestId('editor-activity')).toHaveAttribute('data-hide-no-time', 'true')
    // At the top, before the first card, not with the Activity card at the bottom.
    const following = (later: Element) => notice.compareDocumentPosition(later) & Node.DOCUMENT_POSITION_FOLLOWING
    expect(following(screen.getByRole('textbox', { name: 'Brief' }))).toBeTruthy()
    expect(following(screen.getByTestId('editor-activity'))).toBeTruthy()
    cleanup()

    // Approved with a slot, or scheduled, it says nothing of the kind.
    for (const status of ['approved', 'scheduled'] as const) {
      renderEditor(post({ ...WAITING, status }))
      await screen.findByText('TikTok')
      expect(screen.queryByText(APPROVED_NO_TIME_NOTE, { exact: false }), status).toBeNull()
      cleanup()
    }
  })
})
