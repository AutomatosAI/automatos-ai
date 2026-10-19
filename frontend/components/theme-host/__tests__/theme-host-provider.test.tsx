/**
 * PRD-261 A-01, A-04a — the theme host provider, in every build (no build flag). Under Classic,
 * Studio and builtin Play it renders its children and does nothing else: no load, no global, no
 * status on <html>. Under an installed theme's style it loads that theme, never in safe mode,
 * reports 'loading' from its first render, turns off with no notice when the server has no active
 * install, falls back with a notice, keeps the loaded result as an object (a frame that uses hooks renders
 * cleanly), cleans the theme up when the style changes or it unmounts, and runs loads one after
 * another so a cancelled load cannot strip the live one's look.
 */
import React, { useState } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { THEME_NOTICE_ID } from '@/lib/theme-host/constants'
import type { SlotId } from '@/lib/theme-host/registry-types'
import type { SpikeManifest, ThemeLoadResult } from '@/lib/theme-host/types'
import { UI_STYLES, type UiStyle } from '@/lib/ui-style'

const INSTALLED_STYLE: UiStyle = 'theme:install:00000000-0000-4000-8000-00000000e2e1'
const OTHER_INSTALLED_STYLE: UiStyle = 'theme:install:00000000-0000-4000-8000-000000000000'

const mocks = vi.hoisted(() => ({
  style: { current: 'studio' as string | null },
  loadThemeForStyle: vi.fn(),
  toast: Object.assign(vi.fn(), { error: vi.fn(), dismiss: vi.fn() }),
}))

vi.mock('@/contexts/ui-style-context', () => ({
  useUiStyleOptional: () => (mocks.style.current === null ? null : { style: mocks.style.current }),
}))
vi.mock('sonner', () => ({ toast: mocks.toast }))
vi.mock('@/lib/theme-host/sources', () => ({ loadThemeForStyle: mocks.loadThemeForStyle }))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))

import { PageHeader } from '@/components/shared/page-header'
import { PAGE_HEADER_MASCOT_SLOT, PLACEMENT_SLOT_ATTR } from '../placement-slot'
import { THEME_FALLBACK_NOTICE, THEME_SAFE_MODE_NOTICE, ThemeHostProvider } from '../theme-host-provider'
import { useThemeHost } from '../theme-host-context'

const HOST_GLOBAL = '__AUTOMATOS_THEME_HOST__'
const MANIFEST: SpikeManifest = {
  id: 'automatos-e2e/hello',
  name: 'Hello theme',
  version: '0.0.1',
  sdkRange: '^1.0.0',
  htmlClass: 'hello-spike',
  slots: ['frame@1'],
  entry: 'http://localhost:3000/api/themes/files/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef/entry.js',
  styleSheet: 'http://localhost:3000/api/themes/files/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef/theme.css',
  preset: 'http://localhost:3000/api/themes/files/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef/presets/default.look.json',
}

/** A frame that calls a hook: it renders cleanly only if the provider never passed it to a setter. */
function StatefulFrame({ children }: { children?: React.ReactNode }) {
  const [label] = useState('framed')
  return <div data-testid="f" data-label={label}>{children}</div>
}

function FrameConsumer() {
  const host = useThemeHost()
  const Frame = host?.frame ?? null
  return Frame === null ? <p>no frame</p> : <Frame><p>page</p></Frame>
}

function okResult(cleanup = vi.fn(), slots: ReadonlyMap<SlotId, unknown> = new Map()): ThemeLoadResult {
  return {
    ok: true,
    ref: 'install:00000000-0000-4000-8000-00000000e2e1',
    manifest: MANIFEST,
    frame: StatefulFrame,
    slots,
    pages: new Map(),
    routes: new Map(),
    cleanup,
  }
}

interface Deferred<T> {
  readonly promise: Promise<T>
  readonly resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

/** A load that puts the theme's class on <html> when it starts, as applyLook does; its cleanup removes it. */
function loadThatMarksHtml(pending: Promise<ThemeLoadResult>): () => Promise<ThemeLoadResult> {
  return () => {
    document.documentElement.classList.add(MANIFEST.htmlClass)
    return pending
  }
}

function cleanupThatUnmarksHtml() {
  return vi.fn().mockImplementation(() => document.documentElement.classList.remove(MANIFEST.htmlClass))
}

/** Every status the slot would read, render by render. */
function StatusProbe({ seen }: { seen: string[] }) {
  seen.push(useThemeHost()?.status ?? 'none')
  return null
}

function app() {
  return (
    <ThemeHostProvider>
      <FrameConsumer />
    </ThemeHostProvider>
  )
}

function hostStatus(): string | null {
  return document.documentElement.getAttribute('data-theme-host')
}

beforeEach(() => {
  mocks.style.current = INSTALLED_STYLE
  mocks.loadThemeForStyle.mockReset()
  mocks.toast.mockReset()
  mocks.toast.error.mockReset()
  mocks.toast.dismiss.mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  window.history.replaceState(null, '', '/')
})

describe('PRD-261 A-04a — ThemeHostProvider under a style with no installed theme', () => {
  // The spike's provider wrote 'off' on <html> in its own build; the host now ships in every
  // build, so off leaves no trace (default-build.spec.ts asserts the same in a browser).
  it.each(UI_STYLES)('renders its children and loads, installs and marks nothing under %s', async (style: UiStyle) => {
    mocks.style.current = style
    render(app())
    expect(screen.getByText('no frame')).toBeInTheDocument()
    await act(async () => undefined)
    expect(mocks.loadThemeForStyle).not.toHaveBeenCalled()
    expect(HOST_GLOBAL in window).toBe(false)
    expect(hostStatus()).toBeNull()
    expect(mocks.toast).not.toHaveBeenCalled()
    expect(mocks.toast.error).not.toHaveBeenCalled()
  })

  it('does nothing outside the style provider', async () => {
    mocks.style.current = null
    render(app())
    await act(async () => undefined)
    expect(mocks.loadThemeForStyle).not.toHaveBeenCalled()
    expect(hostStatus()).toBeNull()
  })
})

describe('PRD-261 A-04a — ThemeHostProvider under an installed theme style', () => {
  it('loads the theme of exactly that style', async () => {
    mocks.loadThemeForStyle.mockResolvedValue(okResult())
    render(app())
    await screen.findByTestId('f')
    expect(mocks.loadThemeForStyle).toHaveBeenCalledWith(INSTALLED_STYLE)
  })

  it('loads no theme in safe mode, and says so', async () => {
    window.history.replaceState(null, '', '/agents?safe_mode=themes')
    render(app())
    await waitFor(() => expect(hostStatus()).toBe('safe-mode'))
    expect(mocks.loadThemeForStyle).not.toHaveBeenCalled()
    expect(mocks.toast).toHaveBeenCalledWith(THEME_SAFE_MODE_NOTICE, expect.objectContaining({ id: THEME_NOTICE_ID, duration: Infinity }))
  })

  it('still renders when reading sessionStorage throws', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    mocks.loadThemeForStyle.mockResolvedValue({ ok: false, reason: 'manifest', detail: '404' })
    expect(() => render(app())).not.toThrow()
    await waitFor(() => expect(hostStatus()).toBe('failed'))
  })

  it('falls back with a notice that names the reason', async () => {
    mocks.loadThemeForStyle.mockResolvedValue({ ok: false, reason: 'engines', detail: 'engines.sdk "^2.0.0"' })
    render(app())
    await waitFor(() => expect(hostStatus()).toBe('failed'))
    expect(document.documentElement.getAttribute('data-theme-host-reason')).toBe('engines')
    expect(mocks.toast.error).toHaveBeenCalledWith(THEME_FALLBACK_NOTICE, expect.objectContaining({ id: THEME_NOTICE_ID }))
    expect(screen.getByText('no frame')).toBeInTheDocument()
  })

  it('reports loading from its first render, before its effect starts the load (M69)', async () => {
    const pending = deferred<ThemeLoadResult>()
    mocks.loadThemeForStyle.mockReturnValue(pending.promise)
    const seen: string[] = []
    render(
      <ThemeHostProvider>
        <StatusProbe seen={seen} />
      </ThemeHostProvider>,
    )
    expect(seen[0]).toBe('loading')
    await waitFor(() => expect(hostStatus()).toBe('loading'))
    expect(seen).not.toContain('off')
    // Settle the load, so the module's load queue is free for the next test.
    await act(async () => pending.resolve({ ok: false, reason: 'load', detail: 'settled by the test' }))
    await waitFor(() => expect(hostStatus()).toBe('failed'))
  })

  it('turns off with no notice when the server has no active install (GET /api/themes/active 404)', async () => {
    mocks.loadThemeForStyle.mockResolvedValue({ ok: false, reason: null, detail: 'GET /api/themes/active answered 404: no active install' })
    render(app())
    await waitFor(() => expect(mocks.toast.dismiss).toHaveBeenCalledWith(THEME_NOTICE_ID))
    expect(hostStatus()).toBeNull()
    expect(document.documentElement.getAttribute('data-theme-host-reason')).toBeNull()
    expect(mocks.toast.error).not.toHaveBeenCalled()
    expect(mocks.toast).not.toHaveBeenCalled()
    expect(screen.getByText('no frame')).toBeInTheDocument()
  })

  it('hands the loaded frame to the slot, and a frame that uses hooks renders with no error', async () => {
    const error = vi.spyOn(console, 'error')
    mocks.loadThemeForStyle.mockResolvedValue(okResult())
    render(app())
    expect(await screen.findByTestId('f')).toHaveTextContent('page')
    // The status attribute is set in an effect after the frame commits, so wait for it.
    await waitFor(() => expect(hostStatus()).toBe('active'))
    expect(error).not.toHaveBeenCalled()
  })

  it('cleans the theme up once and turns off, leaving no status, when the style leaves the theme', async () => {
    const cleanup = vi.fn()
    mocks.loadThemeForStyle.mockResolvedValue(okResult(cleanup))
    const { rerender } = render(app())
    await screen.findByTestId('f')
    await waitFor(() => expect(hostStatus()).toBe('active'))
    mocks.style.current = 'studio'
    rerender(app())
    await waitFor(() => expect(hostStatus()).toBeNull())
    expect(cleanup).toHaveBeenCalledTimes(1)
    expect(screen.getByText('no frame')).toBeInTheDocument()
  })

  it('cleans the first theme up and loads the second when the style moves to another install', async () => {
    const cleanup = vi.fn()
    mocks.loadThemeForStyle.mockResolvedValueOnce(okResult(cleanup)).mockResolvedValueOnce(okResult())
    const { rerender } = render(app())
    await screen.findByTestId('f')
    mocks.style.current = OTHER_INSTALLED_STYLE
    rerender(app())
    await waitFor(() => expect(mocks.loadThemeForStyle).toHaveBeenLastCalledWith(OTHER_INSTALLED_STYLE))
    expect(cleanup).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(hostStatus()).toBe('active'))
  })

  it('lets a cancelled load settle and clean up before the next load starts', async () => {
    const first = deferred<ThemeLoadResult>()
    const second = deferred<ThemeLoadResult>()
    mocks.loadThemeForStyle.mockImplementationOnce(loadThatMarksHtml(first.promise)).mockImplementationOnce(loadThatMarksHtml(second.promise))
    const { rerender, unmount } = render(app())
    await waitFor(() => expect(mocks.loadThemeForStyle).toHaveBeenCalledTimes(1))
    mocks.style.current = 'studio'
    rerender(app())
    await waitFor(() => expect(hostStatus()).toBeNull())
    mocks.style.current = INSTALLED_STYLE
    rerender(app())
    await waitFor(() => expect(hostStatus()).toBe('loading'))
    await act(async () => undefined)
    expect(mocks.loadThemeForStyle).toHaveBeenCalledTimes(1)

    const firstCleanup = cleanupThatUnmarksHtml()
    await act(async () => first.resolve(okResult(firstCleanup)))
    await waitFor(() => expect(firstCleanup).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(mocks.loadThemeForStyle).toHaveBeenCalledTimes(2))
    const secondCleanup = cleanupThatUnmarksHtml()
    await act(async () => second.resolve(okResult(secondCleanup)))
    await waitFor(() => expect(hostStatus()).toBe('active'))
    expect(document.documentElement.classList.contains(MANIFEST.htmlClass)).toBe(true)
    expect(secondCleanup).not.toHaveBeenCalled()

    unmount()
    expect(secondCleanup).toHaveBeenCalledTimes(1)
    expect(document.documentElement.classList.contains(MANIFEST.htmlClass)).toBe(false)
  })

  it('cleans the theme up once when it unmounts while active', async () => {
    const cleanup = vi.fn()
    mocks.loadThemeForStyle.mockResolvedValue(okResult(cleanup))
    const { unmount } = render(app())
    await screen.findByTestId('f')
    unmount()
    expect(cleanup).toHaveBeenCalledTimes(1)
    expect(hostStatus()).toBeNull()
  })
})

describe('PRD-261 A-12b — ThemeHostProvider hands the theme\'s placement slots to PageHeader', () => {
  function HeaderMascot({ title }: { title: string }) {
    return <span data-testid="header-mascot">{title}</span>
  }

  function headerApp() {
    return (
      <ThemeHostProvider>
        <FrameConsumer />
        <PageHeader title="Knowledge" titleAccent="Bases" />
      </ThemeHostProvider>
    )
  }

  it('a registered page-header.mascot@1 draws in the header once the theme is active', async () => {
    const slots = new Map<SlotId, unknown>([['frame@1', StatefulFrame], [PAGE_HEADER_MASCOT_SLOT, HeaderMascot]])
    mocks.loadThemeForStyle.mockResolvedValue(okResult(vi.fn(), slots))
    const { container } = render(headerApp())
    expect(await screen.findByTestId('header-mascot')).toHaveTextContent('Knowledge')
    expect(container.querySelector(`[${PLACEMENT_SLOT_ATTR}="${PAGE_HEADER_MASCOT_SLOT}"]`)).toContainElement(screen.getByTestId('header-mascot'))
  })

  it('draws no slot host under a style with no installed theme', async () => {
    mocks.style.current = 'studio'
    const { container } = render(headerApp())
    await act(async () => undefined)
    expect(container.querySelector(`[${PLACEMENT_SLOT_ATTR}]`)).toBeNull()
    expect(mocks.loadThemeForStyle).not.toHaveBeenCalled()
  })
})
