/**
 * Studio's workspace card names the current workspace. MainLayout drew Studio's rail with no name,
 * so the card always read StudioSidebar's default, "Automatos AI", whatever the workspace was. It
 * now passes the workspace provider's name, on a desktop (the Studio shell's rail) and on a phone
 * (the rail in the menu sheet), and the card keeps its default only with no workspace or a blank
 * name. The real StudioSidebar renders; the workspace provider is stubbed.
 */
import React from 'react'
import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

type WorkspaceStub = { workspace: { name: string } | null } | null

const ws = vi.hoisted(() => ({ context: null as WorkspaceStub }))
const viewport = vi.hoisted(() => ({ tabletOrBelow: false }))

vi.mock('next/navigation', () => ({ usePathname: () => '/agents', useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => ws.context }))
vi.mock('../sidebar', () => ({ Sidebar: () => null }))
vi.mock('../mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('../header', () => ({ Header: () => null }))
vi.mock('../studio-header', () => ({ StudioHeader: () => null }))
vi.mock('../../chatbot/chat-widget', () => ({ AutoWidget: () => null }))
// The menu sheet draws its content (the rail) whether or not it is open, so the phone's rail is on screen.
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  SheetContent: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => viewport.tabletOrBelow, useIsMobile: () => false }))
// The Studio style, no theme: the core's chrome.
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useActiveTheme: () => null, useThemeLayout: () => 'fallback' }))

import { MainLayout } from '../main-layout'

const WORKSPACE_NAME = 'Gull & Anchor'
const DEFAULT_NAME = 'Automatos AI'

/** The rail's "Current workspace" card, under MainLayout around a page. */
function renderWorkspaceCard(): HTMLElement {
  render(
    <MainLayout>
      <div data-testid="page" />
    </MainLayout>,
  )
  return screen.getByRole('group', { name: 'Current workspace' })
}

afterEach(() => {
  ws.context = null
  viewport.tabletOrBelow = false
})

describe("Studio's workspace card names the current workspace", () => {
  it.each([
    ['on a desktop (the Studio shell)', false],
    ['on a phone (the rail in the menu sheet)', true],
  ])("reads the workspace provider's name %s, not the default", (_label, phone) => {
    viewport.tabletOrBelow = phone
    ws.context = { workspace: { name: WORKSPACE_NAME } }
    const card = renderWorkspaceCard()
    expect(within(card).getByText(WORKSPACE_NAME)).toBeInTheDocument()
    expect(within(card).getByText('G')).toBeInTheDocument()
    expect(screen.queryByText(DEFAULT_NAME)).toBeNull()
  })

  it.each([
    ['no workspace provider', null],
    ['no workspace loaded yet', { workspace: null }],
    ['a blank name', { workspace: { name: '  ' } }],
  ])('keeps the default with %s', (_label, context) => {
    ws.context = context
    expect(within(renderWorkspaceCard()).getByText(DEFAULT_NAME)).toBeInTheDocument()
  })
})
