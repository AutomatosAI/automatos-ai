/** PRD-244 W3 — the picker carries two axes; picking on one never touches the other. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider } from '@/contexts/ui-style-context'
import { ThemeToggle } from '@/components/ui/theme-toggle'

afterEach(cleanup)

const src = readFileSync(path.resolve(__dirname, '..', 'theme-toggle.tsx'), 'utf8')

describe('ThemeToggle', () => {
  it('mounts as the Appearance control inside the style provider', async () => {
    render(<UiStyleProvider initialStyle="studio"><ThemeToggle /></UiStyleProvider>)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Appearance' })).toBeEnabled())
  })

  it('has a Style group bound to the style context and a Tone group bound to next-themes', () => {
    expect(src.match(/<DropdownMenuRadioGroup/g)).toHaveLength(2)
    expect(src).toContain('value={style}')
    expect(src).toContain('setStyle(v)')
    expect(src).toContain("value={theme ?? 'system'} onValueChange={setTheme}")
    for (const v of ['classic', 'studio', 'light', 'dark', 'system']) expect(src).toContain(`value="${v}"`)
    expect(src).not.toContain("setTheme('studio')")
    expect(src).not.toContain("setStyle('light'")
  })
})
