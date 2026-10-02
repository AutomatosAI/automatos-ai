/**
 * PRD-251 Wave 3, US-305 — a generic channel is labelled "unverified channel" until
 * one of the workspace's targets has published to it; the label the API sends
 * already says so, so the row shows it once, and not at all once it is verified.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { ChannelRow } from '@/components/deliverables/socials/socials-channel-row'
import type { SocialChannel } from '@/lib/api-client'

afterEach(cleanup)

const generic = (verified: boolean): SocialChannel => ({
  toolkit: 'reddit',
  label: verified ? 'Reddit' : 'Reddit (unverified channel)',
  post_kinds: [{ kind: 'image', available: true, reason: null, needs_public_storage: false }],
  verified,
  setup_note: null,
})

function renderRow(channel: SocialChannel) {
  render(
    <ul>
      <ChannelRow channel={channel} kind={null} options={{}} onToggle={vi.fn()} onKind={vi.fn()} onOptions={vi.fn()} />
    </ul>,
  )
}

describe('ChannelRow (socials)', () => {
  it('names an unverified generic channel once', () => {
    renderRow(generic(false))
    expect(screen.getAllByText(/unverified channel/)).toHaveLength(1)
  })

  it('drops the label once the channel has published', () => {
    renderRow(generic(true))
    expect(screen.queryByText(/unverified channel/)).toBeNull()
    expect(screen.getByText('Reddit')).toBeTruthy()
  })
})
