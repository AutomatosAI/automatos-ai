/**
 * PRD-256 F430 — the marketplace detail says what a copy of an agent could not call as a CLI
 * session (`dependencies.session_tool_gaps`), in the local edition, where agents can run as one.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const ITEM = {
  dependencies: {
    session_tool_gaps: [
      { skill: 'quote-cards', tools: ['workspace_html_to_png'], instead: { platform_create_social_post: 'create_social_post' } },
      { skill: 'note-taking', tools: [], instead: { platform_board_summary: 'board_summary' } },
    ],
  },
}

async function load(edition: 'local' | 'saas') {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({ authEdition: edition, isLocal: edition === 'local', isSaaS: edition === 'saas' }))
  vi.doMock('@/lib/api-client', () => ({ apiClient: { request: vi.fn().mockResolvedValue({}) } }))
  return import('../marketplace-session-gaps')
}

afterEach(() => {
  cleanup()
  vi.doUnmock('@/lib/auth-edition')
  vi.doUnmock('@/lib/api-client')
  vi.resetModules()
})

describe('what a CLI-session copy could not call', () => {
  it('lists each skill that names a tool no session has, and leaves out a renamed one', async () => {
    const { cliCopyGapLines } = await load('local')
    const lines = cliCopyGapLines(ITEM)
    expect(lines).toHaveLength(1)
    expect(lines[0]).toContain('`workspace_html_to_png`, which a session cannot use')
    expect(cliCopyGapLines({})).toEqual([])
    expect(cliCopyGapLines({ dependencies: { session_tool_gaps: null } })).toEqual([])
  })

  it('shows the section in the local edition only, and only when there is a gap', async () => {
    const local = await load('local')
    const { rerender } = render(<local.ItemSessionGapsSection item={ITEM} />)
    expect(screen.getByTestId('marketplace-session-gaps').textContent).toContain('As a CLI session')
    rerender(<local.ItemSessionGapsSection item={{ dependencies: {} }} />)
    expect(screen.queryByTestId('marketplace-session-gaps')).not.toBeInTheDocument()
    cleanup()

    const hosted = await load('saas')
    render(<hosted.ItemSessionGapsSection item={ITEM} />)
    expect(screen.queryByTestId('marketplace-session-gaps')).not.toBeInTheDocument()
  })
})
