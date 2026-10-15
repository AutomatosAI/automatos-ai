/**
 * PRD-260 M90 — ⌘K under Play finds Play's own pages: Conversations, Make it yours
 * (/looks) and the style sheet (/looks/style). Under Classic and Studio those pages
 * show only a note and a Switch to Play, so the search doesn't offer them there.
 */
import { describe, it, expect, vi, beforeAll, afterEach } from 'vitest'
import { renderHook, act, render, screen, fireEvent } from '@testing-library/react'

const nav = vi.hoisted(() => ({ push: vi.fn(), play: false }))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: nav.push, replace: vi.fn() }) }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => null }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useIsPlay: () => nav.play }))

import { apiClient } from '@/lib/api-client'
import { useGlobalSearch } from '@/hooks/use-global-search'
import { GlobalSearch } from '@/components/shared/global-search'

function pagesFor(text: string, isPlay: boolean): string[] {
  const { result } = renderHook(() => useGlobalSearch(null, isPlay))
  act(() => result.current.setQuery(text))
  return result.current.pages.map((page) => page.path)
}

afterEach(() => {
  vi.restoreAllMocks()
  nav.push.mockClear()
  nav.play = false
})

describe("global search · Play's pages (hook)", () => {
  it("under Play, 'conv' finds Conversations; under Studio it doesn't", () => {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    expect(pagesFor('conv', true)).toEqual(['/conversations'])
    expect(pagesFor('conv', false)).toEqual([])
  })

  it('under Play, finds Make it yours and the style sheet by their names', () => {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    expect(pagesFor('make it', true)).toEqual(['/looks'])
    expect(pagesFor('style sheet', true)).toEqual(['/looks/style'])
    expect(pagesFor('make it', false)).toEqual([])
    expect(pagesFor('style sheet', false)).toEqual([])
  })

  it('lists them with no query only under Play, after the core pages', () => {
    const play = pagesFor('', true)
    const studio = pagesFor('', false)
    expect(play.slice(-3)).toEqual(['/conversations', '/looks', '/looks/style'])
    expect(play.slice(0, -3)).toEqual(studio)
    expect(studio).not.toContain('/conversations')
  })
})

describe("global search · Play's pages (the dialog reads the style)", () => {
  beforeAll(() => {
    // jsdom has no layout: cmdk's list measures itself, and scrolls its active item into view.
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {}
  })

  async function searchInDialog(text: string) {
    vi.spyOn(apiClient, 'request').mockResolvedValue([] as never)
    render(<GlobalSearch />)
    act(() => {
      window.dispatchEvent(new Event('automatos:global-search-open'))
    })
    fireEvent.change(await screen.findByPlaceholderText(/Search pages/), { target: { value: text } })
  }

  it('under Play, typing "conv" offers Conversations and opens it', async () => {
    nav.play = true
    await searchInDialog('conv')
    fireEvent.click(await screen.findByText('Conversations'))
    expect(nav.push).toHaveBeenCalledWith('/conversations')
  })

  it('under Studio, typing "conv" offers no Conversations page', async () => {
    await searchInDialog('conv')
    expect(await screen.findByPlaceholderText(/Search pages/)).toHaveValue('conv')
    expect(screen.queryByText('Conversations')).toBeNull()
  })
})
