/**
 * PRD-251B US-B301 — Template Studio links to the Brand kit tab (the BrandKitDialog is
 * retired): the gallery's Brand Kit button and the guide go there in place.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const nav = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }))

vi.mock('next/navigation', () => ({
  useRouter: () => nav,
  usePathname: () => '/deliverables',
  useSearchParams: () => new URLSearchParams('tab=templates'),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }))
vi.mock('@/components/documents/blocks/api', () => ({
  templateBlocksApi: {
    listTemplates: vi.fn(async () => []),
    getVariables: vi.fn(async () => ({ variables: [] })),
    listPresets: vi.fn(async () => []),
  },
}))

import { TemplateStudio } from '@/components/documents/blocks/TemplateStudio'

beforeEach(() => {
  nav.push.mockReset()
})
afterEach(cleanup)

describe('Template Studio and the brand kit', () => {
  it("the gallery's Brand Kit goes to the Brand kit tab", async () => {
    render(<TemplateStudio />)
    fireEvent.click(await screen.findByRole('button', { name: 'Brand Kit' }))
    expect(nav.push).toHaveBeenCalledWith('/deliverables?tab=brand')
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
