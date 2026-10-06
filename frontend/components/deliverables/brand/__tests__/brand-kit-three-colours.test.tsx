/**
 * The kit's colours (7 Oct, Gerard): the kit has no accent colour of its own. `accent_color`
 * ("Third colour (social videos)" since F361) is retired: social videos and the social brand
 * board read the palette's accent like documents do, so the page shows three kit colours and
 * points the accent at the Accent role.
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

afterEach(() => cleanup())

describe('the kit colours on the Brand kit page', () => {
  it('shows the primary, the secondary and the body text, and no third colour', () => {
    const { container } = render(<BrandColours kit={designKit()} patch={vi.fn()} />)
    expect(container.querySelectorAll('input[type="color"]')).toHaveLength(3)
    expect(screen.getByLabelText('Primary (brand colour)')).toHaveValue('#c2410c')
    expect(screen.getByLabelText('Secondary')).toHaveValue('#64748b')
    expect(screen.getByLabelText('Body text')).toHaveValue('#1a1a2e')
    expect(screen.queryByLabelText(/third colour/i)).toBeNull()
    expect(screen.getByText(/documents and social posts is the Accent role below/)).toBeInTheDocument()
  })

  it('edits a kit colour', () => {
    const patch = vi.fn()
    render(<BrandColours kit={designKit()} patch={patch} />)
    fireEvent.change(screen.getByLabelText('Secondary'), { target: { value: '#0f5c5c' } })
    expect(patch).toHaveBeenCalledWith({ secondary_color: '#0f5c5c' })
  })
})
