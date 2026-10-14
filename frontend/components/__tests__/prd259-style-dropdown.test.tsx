/**
 * PRD-259 S-004b — the Style dropdown: the Skins group lists the workspace's installed skins,
 * "Add a skin…" posts the file as FormData through apiClient and makes it active (a refused
 * package shows the validator's messages), a dark-only skin disables Light and System, and a
 * skin that isn't installed here falls back with a quiet notice.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { get: api.get, post: api.post } }))
const toasts = vi.hoisted(() => ({ info: vi.fn(), error: vi.fn(), success: vi.fn() }))
vi.mock('sonner', () => ({ toast: toasts }))

import { CASES, SKIN_ID, OTHER_SKIN_ID, SkinHarness, installedItem } from '@/components/__tests__/prd259-skin-harness'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { ADD_SKIN_LABEL } from '@/components/ui/skin-menu-items'
import { IMPORT_REFUSED_TITLE } from '@/hooks/use-skin-import'
import { SKIN_STYLE_ELEMENT_ID } from '@/lib/skins/prepaint'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_SKIN_BASE_COOKIE, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

const html = () => document.documentElement

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
  html().className = ''
  html().removeAttribute('data-skin')
  document.getElementById(SKIN_STYLE_ELEMENT_ID)?.remove()
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  document.cookie = `${UI_SKIN_BASE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  api.get.mockReset()
  api.post.mockReset()
  Object.values(toasts).forEach((fn) => fn.mockClear())
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

async function renderToggle(initialStyle: UiStyle = 'studio') {
  render(<SkinHarness initialStyle={initialStyle} initialSkinBase="studio"><ThemeToggle /></SkinHarness>)
  const trigger = await screen.findByRole('button', { name: 'Appearance' })
  await waitFor(() => expect(trigger).toBeEnabled())
  return trigger
}

async function openMenu(trigger: HTMLElement) {
  act(() => { fireEvent.keyDown(trigger, { key: 'Enter' }) })
  await screen.findByRole('menu')
}

describe('the Style dropdown', () => {
  it('lists the installed skins under a Skins label, after Classic and Studio', async () => {
    api.get.mockResolvedValue([installedItem(CASES.classicLightOnly, SKIN_ID), installedItem(CASES.twoTone, OTHER_SKIN_ID)])
    await openMenu(await renderToggle())
    expect(api.get).toHaveBeenCalledWith('/api/skins/installed')
    await screen.findByRole('menuitemradio', { name: 'Ink' })
    const names = screen.getAllByRole('menuitemradio').map((item) => item.textContent)
    expect(names.slice(0, 4)).toEqual(['Classic', 'Studio', 'Ink', 'Studio, as a skin'])
    expect(screen.getByText('Skins')).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: ADD_SKIN_LABEL })).toBeInTheDocument()
  })

  it('Add a skin… opens the hidden chooser; a file posts as FormData and becomes the active skin', async () => {
    const opened = vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(() => undefined)
    const imported = installedItem(CASES.classicLightOnly, SKIN_ID)
    api.get.mockResolvedValue([])
    api.post.mockResolvedValue(imported)
    const trigger = await renderToggle()
    await openMenu(trigger)
    act(() => { screen.getByRole('menuitem', { name: ADD_SKIN_LABEL }).click() })
    expect(opened).toHaveBeenCalled()
    opened.mockRestore()

    api.get.mockResolvedValue([imported])
    const file = new File(['PK'], 'ink.autoskin', { type: 'application/zip' })
    fireEvent.change(screen.getByTestId('skin-file-input'), { target: { files: [file] } })
    await waitFor(() => expect(html().getAttribute('data-skin')).toBe(SKIN_ID))
    const [pathArg, body] = api.post.mock.calls[0]
    expect(pathArg).toBe('/api/skins/import')
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('file')).toBe(file)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=skin:${SKIN_ID}`)
  })

  it("a refused package shows the validator's messages in a toast", async () => {
    api.get.mockResolvedValue([])
    const detail = { message: "This skin can't be imported.", errors: [
      { code: 'CONTRAST', path: 'tokens.light', message: 'foreground on background is 2.10:1; text needs 4.5:1.' },
      { code: 'UNKNOWN_TOKEN', path: 'tokens.light.glow', message: '"glow" is not a token.' },
    ] }
    api.post.mockRejectedValue(Object.assign(new Error(JSON.stringify(detail)), { status: 400 }))
    await renderToggle()
    fireEvent.change(screen.getByTestId('skin-file-input'), { target: { files: [new File(['x'], 'bad.autoskin')] } })
    await waitFor(() => expect(toasts.error).toHaveBeenCalled())
    const [title, options] = toasts.error.mock.calls[0]
    expect(title).toBe(IMPORT_REFUSED_TITLE)
    expect(options.description).toContain('foreground on background is 2.10:1')
    expect(options.description).toContain('"glow" is not a token.')
    expect(html().hasAttribute('data-skin')).toBe(false)
  })

  it('a dark-only skin locks the tone: Light and System disabled, Dark checked', async () => {
    window.localStorage.setItem('automatos-theme', 'light')
    api.get.mockResolvedValue([installedItem(CASES.darkOnly, SKIN_ID)])
    const trigger = await renderToggle(`skin:${SKIN_ID}`)
    await waitFor(() => expect(html().classList.contains('dark')).toBe(true))
    await openMenu(trigger)
    const tone = (name: string) => screen.getByRole('menuitemradio', { name })
    expect(tone('Light')).toHaveAttribute('aria-disabled', 'true')
    expect(tone('System')).toHaveAttribute('aria-disabled', 'true')
    expect(tone('Dark')).not.toHaveAttribute('aria-disabled')
    expect(tone('Dark')).toHaveAttribute('aria-checked', 'true')
    act(() => { tone('Dark').click() })
    expect(window.localStorage.getItem('automatos-theme')).toBe('light')
  })

  it("a skin that isn't installed here falls back with a quiet notice", async () => {
    api.get.mockResolvedValue([])
    await renderToggle(`skin:${SKIN_ID}`)
    await waitFor(() => expect(toasts.info).toHaveBeenCalledWith("The skin isn't available here, so Studio is showing."))
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=studio`)
  })

  it('ThemeToggle stays one component, with the skin pieces in their own file', () => {
    const src = readFileSync(path.resolve(__dirname, '..', 'ui', 'theme-toggle.tsx'), 'utf8')
    expect(src).toContain('<InstalledSkinItems')
    expect(src).toContain('<AddSkinItem')
    expect(src).not.toMatch(/fetch\(/)
  })
})
