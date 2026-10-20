/**
 * PRD-261 A-12a (PRD §4.4) — the frame renders HeaderUtilities. A theme's frame must keep the host's
 * header utilities (Help and the docs, notifications, the theme menu, the account), so the way back
 * to the theme menu and the account is on screen whatever the theme draws. The SDK's
 * `cards.HeaderUtilities` is that cluster (the Studio header's StudioHeaderUtilities, stood in for
 * here by a marker), and every builtin theme's frame@1 renders it in its header, around the page.
 * An installed theme's frame is checked in the browser lane (e2e/browser/specs/host-cards.spec.ts,
 * fixtures/hello-a12a).
 */
import React from 'react'
import { render, screen, within } from '@testing-library/react'
import { beforeAll, describe, expect, it, vi } from 'vitest'

const UTILITIES = 'host-header-utilities'

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))
// GET /health answers, so the frame shows (the asleep screen has its own tests).
vi.mock('@/hooks/use-backend-health', () => ({
  useBackendHealth: () => ({ connection: 'connected', checking: false, check: vi.fn() }),
}))
// The sidebar has its own tests.
vi.mock('@/components/play/shell/play-sidebar', () => ({ PlaySidebar: () => <aside data-testid="play-sidebar" /> }))
vi.mock('@/components/layout/studio-header', () => ({
  StudioHeaderUtilities: () => <div data-testid="host-header-utilities" />,
}))

import { HeaderUtilities } from '../cards'
import { BUILTIN_THEME_REGISTRATIONS } from '@/lib/theme-host/builtin/slots'

// Builtin Play's frame is a lazy slot: its chunk is fetched once here, so each findBy waits on the
// render, not on the module's first transform.
beforeAll(async () => {
  await import('@/components/play/shell/play-frame')
})

describe('PRD-261 A-12a — the frame renders HeaderUtilities (PRD §4.4)', () => {
  it("the SDK's HeaderUtilities is the host's header utilities cluster", () => {
    render(<HeaderUtilities />)
    expect(screen.getByTestId(UTILITIES)).toBeInTheDocument()
  })

  it('has a builtin theme to check', () => {
    expect(BUILTIN_THEME_REGISTRATIONS.size).toBeGreaterThan(0)
  })

  it.each(Array.from(BUILTIN_THEME_REGISTRATIONS))('%s: frame@1 renders HeaderUtilities in its header, around the page', async (_ref, registrations) => {
    const Frame = registrations.frame
    render(
      <Frame>
        <div data-testid="page" />
      </Frame>,
    )
    const header = await screen.findByRole('banner')
    expect(within(header).getByTestId(UTILITIES)).toBeInTheDocument()
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(header).not.toContainElement(screen.getByTestId('page'))
  })
})
