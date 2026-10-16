/** PRD-260 K4 — toasts follow the tone in effect: the look's forced tone under Play, the stored one otherwise. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render } from '@testing-library/react'

const tone = vi.hoisted(() => ({ theme: 'light', forcedTheme: undefined as string | undefined }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))
const seen = vi.hoisted(() => ({ theme: undefined as string | undefined }))
vi.mock('sonner', () => ({
  Toaster: (props: { theme?: string }) => {
    seen.theme = props.theme
    return null
  },
}))

import { Toaster } from '@/components/ui/sonner'

afterEach(() => {
  cleanup()
  tone.forcedTheme = undefined
  seen.theme = undefined
})

describe('Toaster', () => {
  it('takes the stored tone when nothing forces one', () => {
    render(<Toaster />)
    expect(seen.theme).toBe('light')
  })

  it('takes the tone the look forces under Play', () => {
    tone.forcedTheme = 'dark'
    render(<Toaster />)
    expect(seen.theme).toBe('dark')
  })
})
