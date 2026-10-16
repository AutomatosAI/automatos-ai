/**
 * PRD-261 A-01 — the theme host provider. With the spike off it renders its children and does
 * nothing else. With it on, it loads the theme only while the style is exactly Studio, never in
 * safe mode, falls back with a notice, keeps the loaded result as an object (a frame that uses
 * hooks renders cleanly), and cleans the theme up when the style changes or it unmounts.
 */
import React, { useState } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SPIKE_BASE_STYLE, THEME_NOTICE_ID } from '@/lib/theme-host/constants'
import type { SpikeManifest, ThemeLoadResult } from '@/lib/theme-host/types'
import { UI_STYLES, type UiStyle } from '@/lib/ui-style'

const mocks = vi.hoisted(() => ({
  style: { current: 'studio' as string },
  loadSpikeTheme: vi.fn(),
  toast: Object.assign(vi.fn(), { error: vi.fn(), dismiss: vi.fn() }),
}))

vi.mock('@/contexts/ui-style-context', () => ({
  useUiStyleOptional: () => ({ style: mocks.style.current }),
}))
vi.mock('sonner', () => ({ toast: mocks.toast }))
vi.mock('@/lib/theme-host/loader', () => ({ loadSpikeTheme: mocks.loadSpikeTheme }))

import { THEME_FALLBACK_NOTICE, THEME_SAFE_MODE_NOTICE, ThemeHostProvider } from '../theme-host-provider'
import { useThemeHost } from '../theme-host-context'

const SPIKE_FLAG = 'AUTOMATOS_THEME_HOST_SPIKE'
const HOST_GLOBAL = '__AUTOMATOS_THEME_HOST__'
const MANIFEST: SpikeManifest = {
  id: 'automatos-e2e/hello',
  name: 'Hello theme',
  version: '0.0.1',
  sdkRange: '^1.0.0',
  htmlClass: 'hello-spike',
  slots: ['frame@1'],
  entry: 'http://localhost:3000/theme-spike/hello/entry.js',
  styleSheet: 'http://localhost:3000/theme-spike/hello/theme.css',
  preset: 'http://localhost:3000/theme-spike/hello/presets/default.look.json',
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

function okResult(cleanup = vi.fn()): ThemeLoadResult {
  return { ok: true, manifest: MANIFEST, frame: StatefulFrame, cleanup }
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
  mocks.style.current = SPIKE_BASE_STYLE
  mocks.loadSpikeTheme.mockReset()
  mocks.toast.mockReset()
  mocks.toast.error.mockReset()
  mocks.toast.dismiss.mockReset()
})

afterEach(() => {
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  window.history.replaceState(null, '', '/')
})

describe('PRD-261 A-01 — ThemeHostProvider with the spike off', () => {
  it('renders its children and loads, installs and marks nothing', async () => {
    vi.stubEnv(SPIKE_FLAG, '0')
    render(app())
    expect(screen.getByText('no frame')).toBeInTheDocument()
    await act(async () => undefined)
    expect(mocks.loadSpikeTheme).not.toHaveBeenCalled()
    expect(HOST_GLOBAL in window).toBe(false)
    expect(hostStatus()).toBeNull()
    expect(mocks.toast).not.toHaveBeenCalled()
  })
})

describe('PRD-261 A-01 — ThemeHostProvider with the spike on', () => {
  beforeEach(() => {
    vi.stubEnv(SPIKE_FLAG, '1')
  })

  it.each(UI_STYLES.filter((style) => style !== SPIKE_BASE_STYLE))('loads no theme under the %s style', async (style: UiStyle) => {
    mocks.style.current = style
    render(app())
    await waitFor(() => expect(hostStatus()).toBe('off'))
    await act(async () => undefined)
    expect(mocks.loadSpikeTheme).not.toHaveBeenCalled()
  })

  it('loads no theme in safe mode, and says so', async () => {
    window.history.replaceState(null, '', '/agents?safe_mode=themes')
    render(app())
    await waitFor(() => expect(hostStatus()).toBe('safe-mode'))
    expect(mocks.loadSpikeTheme).not.toHaveBeenCalled()
    expect(mocks.toast).toHaveBeenCalledWith(THEME_SAFE_MODE_NOTICE, expect.objectContaining({ id: THEME_NOTICE_ID, duration: Infinity }))
  })

  it('still renders when reading sessionStorage throws', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    mocks.loadSpikeTheme.mockResolvedValue({ ok: false, reason: 'manifest', detail: '404' })
    expect(() => render(app())).not.toThrow()
    await waitFor(() => expect(hostStatus()).toBe('failed'))
  })

  it('falls back with a notice that names the reason', async () => {
    mocks.loadSpikeTheme.mockResolvedValue({ ok: false, reason: 'engines', detail: 'engines.sdk "^2.0.0"' })
    render(app())
    await waitFor(() => expect(hostStatus()).toBe('failed'))
    expect(document.documentElement.getAttribute('data-theme-host-reason')).toBe('engines')
    expect(mocks.toast.error).toHaveBeenCalledWith(THEME_FALLBACK_NOTICE, expect.objectContaining({ id: THEME_NOTICE_ID }))
    expect(screen.getByText('no frame')).toBeInTheDocument()
  })

  it('hands the loaded frame to the slot, and a frame that uses hooks renders with no error', async () => {
    const error = vi.spyOn(console, 'error')
    mocks.loadSpikeTheme.mockResolvedValue(okResult())
    render(app())
    expect(await screen.findByTestId('f')).toHaveTextContent('page')
    expect(hostStatus()).toBe('active')
    expect(error).not.toHaveBeenCalled()
  })

  it('cleans the theme up once and turns off when the style leaves Studio', async () => {
    const cleanup = vi.fn()
    mocks.loadSpikeTheme.mockResolvedValue(okResult(cleanup))
    const { rerender } = render(app())
    await screen.findByTestId('f')
    mocks.style.current = UI_STYLES.find((style) => style !== SPIKE_BASE_STYLE) ?? 'classic'
    rerender(app())
    await waitFor(() => expect(hostStatus()).toBe('off'))
    expect(cleanup).toHaveBeenCalledTimes(1)
    expect(screen.getByText('no frame')).toBeInTheDocument()
  })

  it('cleans the theme up once when it unmounts while active', async () => {
    const cleanup = vi.fn()
    mocks.loadSpikeTheme.mockResolvedValue(okResult(cleanup))
    const { unmount } = render(app())
    await screen.findByTestId('f')
    unmount()
    expect(cleanup).toHaveBeenCalledTimes(1)
    expect(hostStatus()).toBeNull()
  })
})
