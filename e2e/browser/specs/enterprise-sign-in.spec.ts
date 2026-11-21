import { expect, test } from '@playwright/test'

// PRD-277 US-502 (@enterprise): signing in with the organisation's identity provider, end to end.
//
//   a signed-out visit to a protected page → /sign-in?return_to=<page> → one button, "Sign in with
//   <OIDC_DISPLAY_NAME>", and no password field → the API's start route → the identity provider
//   (the test issuer approves at once) → the API's callback sets the session cookie → back on the
//   page asked for, signed in → Sign out → the API ends the session and sends the browser to the
//   provider's end-session address → the same page now sends the browser to sign in again.
//
// It needs a stack the lane's local-edition job does not start, so it runs only where
// E2E_ENTERPRISE_APP_URL names one (skipped otherwise):
//   - the frontend built with NEXT_PUBLIC_AUTH_EDITION=enterprise, and NEXT_PUBLIC_API_URL its own
//     origin: the page and /api must share one address (PRD-277 §5.3), so a reverse proxy sends
//     /api/v1/… and the API's other routes to the backend, and everything else to the frontend;
//   - the backend with AUTH_EDITION=enterprise, FRONTEND_URL that origin, OIDC_ISSUER,
//     OIDC_CLIENT_ID and the client secret pointing at the test issuer, and OIDC_DISPLAY_NAME;
//   - the test issuer (orchestrator/tests/helpers_fake_oidc_provider.py, FakeIdentityProvider):
//     a loopback TLS server with a certificate of its own, so the browser context ignores HTTPS
//     errors. Its /authorize approves at once; it has no /logout page, so the end-session address
//     answers 404, which is enough to show the browser was sent there.

const APP_URL = process.env.E2E_ENTERPRISE_APP_URL ?? ''
const DISPLAY_NAME = process.env.E2E_ENTERPRISE_DISPLAY_NAME ?? 'your organisation'
const PROTECTED_PAGE = '/settings/profile'
const SIGN_IN_PATH = '/sign-in'
const SESSION_COOKIE = '__Host-automatos_session'
const PROFILE_NOTE = /managed by your organisation's identity provider/i
// The header's profile menu (frontend/components/auth/profile-menu.tsx), shown in enterprise too.
const USER_MENU = 'User menu'
const LOGOUT = 'Logout'

test.describe('@enterprise sign-in with the organisation\'s identity provider', () => {
  test.skip(!APP_URL, 'E2E_ENTERPRISE_APP_URL names no enterprise stack')
  test.use({ baseURL: APP_URL, ignoreHTTPSErrors: true })

  test('signs in, opens a protected page, and signs out', async ({ page, context }) => {
    await page.goto(PROTECTED_PAGE)
    await expect(page).toHaveURL((url) => url.pathname === SIGN_IN_PATH && url.searchParams.get('return_to') === PROTECTED_PAGE)
    await expect(page.locator('input[type="password"]')).toHaveCount(0)
    const signIn = page.getByRole('link', { name: `Sign in with ${DISPLAY_NAME}` })
    await expect(signIn).toBeVisible()

    await signIn.click()
    await expect(page).toHaveURL((url) => url.pathname === PROTECTED_PAGE)
    await expect(page.getByText(PROFILE_NOTE).first()).toBeVisible()
    const cookies = await context.cookies()
    const session = cookies.find((cookie) => cookie.name === SESSION_COOKIE)
    expect(session?.httpOnly).toBe(true)
    expect(session?.secure).toBe(true)
    expect(await page.evaluate(() => document.cookie)).not.toContain(SESSION_COOKIE)

    await page.getByRole('button', { name: USER_MENU }).first().click()
    await page.getByRole('menuitem', { name: LOGOUT }).click()
    await expect(page).not.toHaveURL((url) => url.origin === new URL(APP_URL).origin && url.pathname === PROTECTED_PAGE)
    expect((await context.cookies()).some((cookie) => cookie.name === SESSION_COOKIE)).toBe(false)

    await page.goto(PROTECTED_PAGE)
    await expect(page).toHaveURL((url) => url.pathname === SIGN_IN_PATH)
  })
})
