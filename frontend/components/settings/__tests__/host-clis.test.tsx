/**
 * PRD-253 S3.1 — a host's line in Settings names every CLI it announced: served,
 * or why not, and for GitHub Copilot the account whose seat the tickets spend.
 */
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'

import { hostCliLines } from '@/components/settings/host-clis'
import { HostsList, type HostRow } from '@/components/settings/SessionModeSections'

const clis = {
  claude: { version: '2.1.236 (Claude Code)', served: true },
  codex: { version: null, served: false, reason: 'Codex is not installed on this machine' },
  copilot: { version: '1.0.91', served: true, login: 'octocat@github.com', login_route: 'copilot' },
}

describe('hostCliLines', () => {
  it('names each CLI with its version, who it runs as, or why it is not served', () => {
    expect(hostCliLines(clis)).toEqual([
      { id: 'claude', served: true, text: 'Claude Code · 2.1.236 (Claude Code)' },
      { id: 'codex', served: false, text: 'Codex — Codex is not installed on this machine' },
      { id: 'copilot', served: true, text: 'GitHub Copilot · 1.0.91 — runs as octocat@github.com (Copilot login)' },
    ])
    expect(hostCliLines({ copilot: { served: true, login: 'me@github.com', login_route: 'gh' } })[0].text)
      .toBe('GitHub Copilot — runs as me@github.com (GitHub CLI login)')
    expect(hostCliLines(null)).toEqual([])
  })
})

describe('HostsList', () => {
  it('lists the three CLIs under the host', () => {
    const host: HostRow = { id: 'abcdef123456', name: 'laptop', status: 'paired', online: true, capabilities: { clis: clis, providers: ['claude', 'copilot'] } }
    render(<HostsList hosts={[host]} />)
    const list = screen.getByTestId('host-clis-abcdef123456')
    expect(list.querySelectorAll('li')).toHaveLength(3)
    expect(list.textContent).toContain('runs as octocat@github.com')
    expect(list.textContent).toContain('not installed')
  })
})
