/**
 * PRD-261 A-04c — /safe-mode turns safe mode on for this tab and links back into the app, still
 * in safe mode. It stands alone: no MainLayout, no backend call.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// A plain anchor that takes the ref Button's asChild slot hands it.
vi.mock('next/link', async () => {
  const { forwardRef } = await import('react')
  return {
    default: forwardRef<HTMLAnchorElement, React.AnchorHTMLAttributes<HTMLAnchorElement>>(function MockLink(
      { href, children, ...rest },
      ref,
    ) {
      return (
        <a ref={ref} href={href} {...rest}>
          {children}
        </a>
      )
    }),
  }
})
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))

import { SAFE_MODE_STORAGE_KEY } from '@/lib/theme-host/constants'
import { isSafeModeRequested, safeSessionStorage } from '@/lib/theme-host/safe-mode'
import SafeModePage from '../page'

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
})

describe('PRD-261 A-04c — the /safe-mode page', () => {
  it('turns safe mode on for this tab', () => {
    render(<SafeModePage />)
    expect(window.sessionStorage.getItem(SAFE_MODE_STORAGE_KEY)).toBe('1')
    // A later page of the tab, with no query, is in safe mode too.
    expect(isSafeModeRequested('', safeSessionStorage())).toBe(true)
  })

  it('links back into the app, still asking for safe mode', () => {
    render(<SafeModePage />)
    const back = screen.getByTestId('safe-mode-back')
    expect(back.tagName).toBe('A')
    expect(back).toHaveAttribute('href', '/chat?safe_mode=themes')
    expect(back).toHaveTextContent('Back to the app')
  })

  it('says what safe mode does, under a page header', () => {
    render(<SafeModePage />)
    expect(screen.getByText(/Themes are off in this tab/)).toBeInTheDocument()
    expect(screen.getByText(/lasts until you close this tab/)).toBeInTheDocument()
  })

  it('still renders where sessionStorage cannot be read', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    expect(() => render(<SafeModePage />)).not.toThrow()
    expect(screen.getByTestId('safe-mode-back')).toHaveAttribute('href', '/chat?safe_mode=themes')
  })

  it('calls no backend route', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    render(<SafeModePage />)
    await Promise.resolve()
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})
