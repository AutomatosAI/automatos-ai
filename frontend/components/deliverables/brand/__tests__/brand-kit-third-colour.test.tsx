/**
 * F361 (night 10c) — one accent on the Brand kit page. The kit's `accent_color` was labelled
 * "Accent" (navy) above the role "Accent (highlights)" (orange); documents never draw it as
 * their accent. It is the "Third colour (social videos)" now, so no kit colour is called an
 * accent: the accent is the highlight role under Colours.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

// The fields' module loads the font uploader, which reaches the backend only through apiClient.
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    get: vi.fn(), put: vi.fn(), post: vi.fn(), delete: vi.fn(),
    getAuthHeaders: vi.fn(async () => ({})), getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { BrandColours } from '../brand-kit-fields'
import { designKit } from './brand-kit-design-kit'

const THIRD = '#1d3658'

afterEach(() => cleanup())

describe('the kit colours on the Brand kit page (F361)', () => {
  it('labels accent_color the third colour, never "Accent"', () => {
    render(<BrandColours kit={designKit({ accent_color: THIRD })} patch={vi.fn()} />)
    expect(screen.getByLabelText('Third colour (social videos)')).toHaveValue(THIRD)
    expect(screen.queryByText(/accent/i, { selector: 'label' })).toBeNull()
    expect(screen.getByText(/highlight colour in\s+documents is the Accent role below/)).toBeInTheDocument()
  })

  it('still edits accent_color', () => {
    const patch = vi.fn()
    render(<BrandColours kit={designKit({ accent_color: THIRD })} patch={patch} />)
    fireEvent.change(screen.getByLabelText('Third colour (social videos)'), { target: { value: '#0f5c5c' } })
    expect(patch).toHaveBeenCalledWith({ accent_color: '#0f5c5c' })
  })
})
