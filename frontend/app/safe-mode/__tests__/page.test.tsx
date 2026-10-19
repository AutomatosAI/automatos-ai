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

import {
  SAFE_MODE_ADD_CLASSES,
  SAFE_MODE_REMOVE_CLASSES,
  SAFE_MODE_UNPAINT_ID,
  SAFE_MODE_UNPAINT_SCRIPT,
} from '@/components/theme-host/theme-host-provider'
import { THEME_BOOT_ELEMENT_ID } from '@/lib/theme-host/boot'
import { SAFE_MODE_STORAGE_KEY } from '@/lib/theme-host/constants'
import { isSafeModeRequested, safeSessionStorage } from '@/lib/theme-host/safe-mode'
import SafeModePage from '../page'

/** The installed theme's own class and a prepaint variable, as the root layout's boot leaves them. */
const THEME_CLASS = 'hello-a04c'
const PREPAINT_VAR = '--background'

/** <html> as THEME_BOOT_SCRIPT leaves it for an installed theme: its classes, another style's, and its prepaint. */
function paintInstalledThemeBoot(): void {
  const boot = document.createElement('script')
  boot.type = 'application/json'
  boot.id = THEME_BOOT_ELEMENT_ID
  boot.textContent = JSON.stringify({ addClasses: [...SAFE_MODE_ADD_CLASSES, THEME_CLASS] })
  document.head.appendChild(boot)
  const root = document.documentElement
  root.className = ''
  root.classList.add(THEME_CLASS, ...SAFE_MODE_REMOVE_CLASSES)
  root.style.setProperty(PREPAINT_VAR, '0 0% 0%')
}

function expectStudioPaint(): void {
  const root = document.documentElement
  SAFE_MODE_ADD_CLASSES.forEach((name) => expect(root.classList.contains(name)).toBe(true))
  ;[THEME_CLASS, ...SAFE_MODE_REMOVE_CLASSES].forEach((name) => expect(root.classList.contains(name)).toBe(false))
  expect(root.style.getPropertyValue(PREPAINT_VAR)).toBe('')
}

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  document.getElementById(THEME_BOOT_ELEMENT_ID)?.remove()
  document.documentElement.className = ''
  document.documentElement.removeAttribute('style')
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

  it('takes the installed theme\'s first paint off <html> and shows Studio\'s', () => {
    expect(SAFE_MODE_ADD_CLASSES.length).toBeGreaterThan(0)
    expect(SAFE_MODE_REMOVE_CLASSES.length).toBeGreaterThan(0)
    paintInstalledThemeBoot()
    render(<SafeModePage />)
    expectStudioPaint()
  })

  it('puts the constant script first in its HTML, and it does the same before the page paints', () => {
    const { container } = render(<SafeModePage />)
    const script = container.querySelector('main')?.firstElementChild
    expect(script?.tagName).toBe('SCRIPT')
    expect(script).toHaveAttribute('id', SAFE_MODE_UNPAINT_ID)
    expect(script?.innerHTML).toBe(SAFE_MODE_UNPAINT_SCRIPT)
    expect(script).toHaveAttribute('data-add', SAFE_MODE_ADD_CLASSES.join(' '))
    expect(script).toHaveAttribute('data-remove', SAFE_MODE_REMOVE_CLASSES.join(' '))
    // The script as the browser runs it while parsing, with the page's own element in place.
    paintInstalledThemeBoot()
    new Function(SAFE_MODE_UNPAINT_SCRIPT)()
    expectStudioPaint()
  })

  it('still takes every other style\'s class off where the boot element cannot be read', () => {
    render(<SafeModePage />)
    const boot = document.createElement('script')
    boot.id = THEME_BOOT_ELEMENT_ID
    // A JSON element, like the real boot element: without the type, jsdom would run the text as code.
    boot.type = 'application/json'
    boot.textContent = '{not json'
    document.head.appendChild(boot)
    document.documentElement.classList.add(...SAFE_MODE_REMOVE_CLASSES)
    expect(() => new Function(SAFE_MODE_UNPAINT_SCRIPT)()).not.toThrow()
    SAFE_MODE_REMOVE_CLASSES.forEach((name) => expect(document.documentElement.classList.contains(name)).toBe(false))
  })

  it('calls no backend route', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    render(<SafeModePage />)
    await Promise.resolve()
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})
