/**
 * An outside link (a mail tool's attachment URL) is opened only when it is http(s), and the
 * page it opens gets no window.opener. A `javascript:` URL opened with window.open would run
 * with the app's origin.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { EXTERNAL_WINDOW_FEATURES, externalHttpUrl, openExternalUrl } from '@/lib/external-url'

afterEach(() => vi.restoreAllMocks())

describe('externalHttpUrl', () => {
  it.each([
    ['https://graph.microsoft.com/v1.0/me/messages/1/attachments/2', 'https://graph.microsoft.com/v1.0/me/messages/1/attachments/2'],
    ['http://files.example/a.pdf', 'http://files.example/a.pdf'],
    ['  https://files.example/a.pdf ', 'https://files.example/a.pdf'],
  ])('keeps an http(s) link: %s', (given, kept) => {
    expect(externalHttpUrl(given)).toBe(kept)
  })

  it.each([
    'javascript:alert(document.cookie)',
    ' JaVaScRiPt:alert(1)',
    'data:text/html,<script>x()</script>',
    'vbscript:msgbox(1)',
    'file:///etc/passwd',
    '/api/attachments/1',
    'not a url',
    '',
    null,
    undefined,
    42,
  ])('refuses %s', (given) => {
    expect(externalHttpUrl(given)).toBeNull()
  })
})

describe('openExternalUrl', () => {
  it('opens an http(s) link in a new tab with no opener', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    expect(openExternalUrl('https://files.example/a.pdf')).toBe(true)
    expect(open).toHaveBeenCalledWith('https://files.example/a.pdf', '_blank', EXTERNAL_WINDOW_FEATURES)
    expect(EXTERNAL_WINDOW_FEATURES).toContain('noopener')
  })

  it('never opens a script link', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    expect(openExternalUrl('javascript:alert(1)')).toBe(false)
    expect(open).not.toHaveBeenCalled()
  })
})
