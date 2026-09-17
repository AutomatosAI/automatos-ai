/** PRD-244 W2 — /missions forwards to the Assignments Missions tab with every other query param kept, at every width. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, cleanup, waitFor } from '@testing-library/react'

const nav = vi.hoisted(() => ({ replace: vi.fn(), search: 'state=awaiting_approval&x=1' }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))

import MissionsRoute from '@/app/missions/page'

afterEach(() => { cleanup(); nav.replace.mockClear() })

describe('/missions redirect', () => {
  it('keeps the other query params and pre-selects the Missions tab', async () => {
    render(<MissionsRoute />)
    await waitFor(() => expect(nav.replace).toHaveBeenCalled())
    const target = String(nav.replace.mock.calls[0][0])
    expect(target.startsWith('/assignments?')).toBe(true)
    const params = new URLSearchParams(target.split('?')[1])
    expect(params.get('tab')).toBe('missions')
    expect(params.get('state')).toBe('awaiting_approval')
    expect(params.get('x')).toBe('1')
  })
})
