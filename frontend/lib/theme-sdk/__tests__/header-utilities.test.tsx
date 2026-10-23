/**
 * PRD-261 A-12a (PRD §4.4) — the frame renders HeaderUtilities. A theme's frame must keep the host's
 * header utilities (Help and the docs, notifications, the theme menu, the account), so the way back
 * to the theme menu and the account is on screen whatever the theme draws. The SDK's
 * `cards.HeaderUtilities` is that cluster (the Studio header's StudioHeaderUtilities, stood in for
 * here by a marker), and it is what a theme takes from '@automatos/theme-sdk'.
 *
 * No theme is compiled into the core (A-19f1), so the frame here is an installed theme's: the neutral
 * fixture theme (lib/theme-host/__tests__/fixture-theme-entry.ts) with a frame@1 drawn from the SDK
 * alone, as the host-cards hello theme's is (e2e/browser/fixtures/hello-a12a/entry.js, held to
 * rendering HeaderUtilities by hello-a12a-fixture.test.ts), under the host's frame slot. The real
 * installed frame is checked in the browser lane (e2e/browser/specs/host-cards.spec.ts).
 */
import React from 'react'
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const UTILITIES = 'host-header-utilities'

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))
vi.mock('@/components/layout/studio-header', () => ({
  StudioHeaderUtilities: () => <div data-testid="host-header-utilities" />,
}))

import * as ThemeSdk from '@automatos/theme-sdk'
import { ThemeFrameSlot } from '@/components/theme-host/theme-frame-slot'
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { fixtureThemeHost } from '@/lib/theme-host/__tests__/fixture-theme-entry'
import type { FrameProps } from '@/lib/theme-host/types'

import { HeaderUtilities } from '../cards'

const FALLBACK = 'core-chrome'

/** An installed theme's frame@1, from the SDK alone: its own header around the host's cluster, then the page. */
function HeaderFrame({ children }: FrameProps) {
  return (
    <div data-testid="theme-frame">
      <header>
        <strong>Fixture theme</strong>
        <ThemeSdk.HeaderUtilities />
      </header>
      <main>{children}</main>
    </div>
  )
}

describe('PRD-261 A-12a — the frame renders HeaderUtilities (PRD §4.4)', () => {
  it("the SDK's HeaderUtilities is the host's header utilities cluster", () => {
    render(<HeaderUtilities />)
    expect(screen.getByTestId(UTILITIES)).toBeInTheDocument()
  })

  it("is the HeaderUtilities a theme imports from '@automatos/theme-sdk'", () => {
    expect(ThemeSdk.HeaderUtilities).toBe(HeaderUtilities)
  })

  it("an installed theme's frame@1 renders it in its header, around the page, under the host's frame slot", () => {
    render(
      <ThemeHostContext.Provider value={fixtureThemeHost('active', { frame: HeaderFrame })}>
        <ThemeFrameSlot fallback={<div data-testid={FALLBACK} />}>
          <div data-testid="page" />
        </ThemeFrameSlot>
      </ThemeHostContext.Provider>,
    )
    const header = screen.getByRole('banner')
    expect(within(screen.getByTestId('theme-frame')).getByRole('banner')).toBe(header)
    expect(within(header).getByTestId(UTILITIES)).toBeInTheDocument()
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(header).not.toContainElement(screen.getByTestId('page'))
    expect(screen.queryByTestId(FALLBACK)).toBeNull()
  })
})
