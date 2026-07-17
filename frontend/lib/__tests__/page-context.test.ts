import { describe, it, expect, vi } from 'vitest'
import { renderHook } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

// PRD-221 S5 — usePageContext builds the structured reference set from the
// route, and BOTH chat surfaces send it. The dead userRole field is gone.

let mockPath = '/command-center'
vi.mock('next/navigation', () => ({ usePathname: () => mockPath }))

import { usePageContext } from '@/lib/page-context'

describe('usePageContext', () => {
  it('resolves the manifest key and route from the pathname', () => {
    mockPath = '/command-center'
    const { result } = renderHook(() => usePageContext())
    expect(result.current.page).toBe('command-center')
    expect(result.current.route).toBe('/command-center')
  })

  it('resolves subpaths to their owning page', () => {
    mockPath = '/missions/abc-123'
    const { result } = renderHook(() => usePageContext())
    expect(result.current.page).toBe('missions')
  })

  it('returns page "unknown" for an unmapped route', () => {
    mockPath = '/nope'
    const { result } = renderHook(() => usePageContext())
    expect(result.current.page).toBe('unknown')
  })

  it('merges page-owned extras like tab and selected', () => {
    mockPath = '/command-center'
    const { result } = renderHook(() =>
      usePageContext({ tab: 'watchlist', selected: { type: 'watch', id: 'w_1' } }),
    )
    expect(result.current.tab).toBe('watchlist')
    expect(result.current.selected).toEqual({ type: 'watch', id: 'w_1' })
  })
})

describe('page context wiring (source contracts)', () => {
  const read = (p: string) => readFileSync(path.resolve(__dirname, '..', '..', p), 'utf8')

  it('the widget sends structured pageContext, not a label string', () => {
    const src = read('components/chatbot/chat-widget.tsx')
    expect(src).toContain('usePageContext')
    expect(src).toContain('pageContext')
    expect(src).not.toContain('pageContext: pageLabel')
  })

  it('the main chat sends page context too', () => {
    const src = read('components/chatbot/chat.tsx')
    expect(src).toContain('usePageContext')
    expect(src).toContain('pageContext')
  })

  it('the request context type no longer carries userRole', () => {
    const src = read('types/chat.ts')
    expect(src).not.toContain('userRole')
  })

  it('hooks.ts forwards the whole structured context object', () => {
    const src = read('lib/chat/hooks.ts')
    expect(src).toContain('context: pageContext')
    expect(src).not.toContain('context: { page: pageContext }')
  })
})
