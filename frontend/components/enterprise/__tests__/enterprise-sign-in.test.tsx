/**
 * PRD-277 US-502 — the enterprise sign-in page.
 *
 *   - One button, "Sign in with <OIDC_DISPLAY_NAME>" (GET /api/v1/auth/sign-in-page), that goes to
 *     the API's start route with `return_to` passed through untouched: the API's rule decides it.
 *   - No password field, no sign-up, no password reset.
 *   - `?error=<code>` shows that code's one sentence; an unknown code the general one.
 *   - The page under app/sign-in renders it in the enterprise build (not Clerk's <SignIn>).
 */
import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }))
const api = vi.hoisted(() => ({ get: vi.fn() }))

vi.mock('next/navigation', () => ({
  useSearchParams: () => nav.params,
  redirect: vi.fn(),
}))
vi.mock('@/lib/api-client', () => ({ apiClient: { get: api.get, getBaseUrl: () => '' } }))

import { EnterpriseSignIn } from '@/components/enterprise/enterprise-sign-in'
import { SIGN_IN_REFUSALS, SIGN_IN_REFUSAL_FALLBACK } from '@/lib/enterprise/messages'

afterEach(() => {
  nav.params = new URLSearchParams()
  api.get.mockReset()
})

async function shown(query = '') {
  nav.params = new URLSearchParams(query)
  api.get.mockResolvedValue({ display_name: 'Example Corp' })
  render(<EnterpriseSignIn />)
  return screen.findByRole('link', { name: 'Sign in with Example Corp' })
}

describe('PRD-277 US-502 — the enterprise sign-in page', () => {
  it("has one button, named for the organisation's provider, that starts the sign-in", async () => {
    const button = await shown()
    expect(api.get).toHaveBeenCalledWith('/api/v1/auth/sign-in-page')
    expect(button.getAttribute('href')).toBe('/api/v1/auth/oidc/start')
    expect(screen.getAllByRole('link')).toHaveLength(1)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('passes where the person was going through to the start route, untouched', async () => {
    const button = await shown('return_to=%2Fchat%2Fabc%3Ftab%3Dfiles')
    const target = new URL(button.getAttribute('href') ?? '', 'https://automatos.example.test')
    expect(target.pathname).toBe('/api/v1/auth/oidc/start')
    expect(target.searchParams.get('return_to')).toBe('/chat/abc?tab=files')
  })

  it('has no password field, no sign-up and no reset', async () => {
    const { container } = render(<EnterpriseSignIn />)
    await waitFor(() => expect(container.querySelector('a')).not.toBeNull())
    expect(container.querySelector('input')).toBeNull()
    expect(screen.queryByText(/sign up|forgot|reset/i)).toBeNull()
  })

  it.each(Object.keys(SIGN_IN_REFUSALS))('shows the sentence for ?error=%s', async (code) => {
    await shown(`error=${code}`)
    expect(screen.getByRole('alert').textContent).toBe(SIGN_IN_REFUSALS[code])
  })

  it('shows the general sentence for a code it does not know, never the code', async () => {
    await shown('error=%3Cb%3Ehello%3C%2Fb%3E')
    expect(screen.getByRole('alert').textContent).toBe(SIGN_IN_REFUSAL_FALLBACK)
  })

  it('falls back to the default name when the API cannot be asked', async () => {
    api.get.mockRejectedValue(new Error('down'))
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    render(<EnterpriseSignIn />)
    expect(await screen.findByRole('link', { name: 'Sign in with your organisation' })).toBeTruthy()
  })
})

describe('app/sign-in in the enterprise build', () => {
  it("renders the enterprise page, not Clerk's", async () => {
    vi.resetModules()
    vi.doMock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
    vi.doMock('@/lib/auth-edition', () => ({ isSaaS: false }))
    vi.doMock('@clerk/nextjs', () => ({
      SignIn: () => {
        throw new Error('Clerk rendered')
      },
    }))
    api.get.mockResolvedValue({ display_name: 'Example Corp' })
    const { default: SignInPage } = await import('@/app/sign-in/[[...rest]]/page')
    render(<SignInPage />)
    expect(await screen.findByRole('link', { name: 'Sign in with Example Corp' })).toBeTruthy()
  })
})
