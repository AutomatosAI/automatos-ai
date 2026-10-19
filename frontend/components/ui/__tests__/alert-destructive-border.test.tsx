/**
 * PRD-260 P260-FX-05 (M41, review #179) — a destructive Alert keeps the border its caller asks for.
 *
 * The destructive variant carried `dark:border-destructive`, dead until PRD-260's darkMode
 * cherry-pick made every dark: class render. tailwind-merge drops `border-destructive/50`
 * for a caller's `border-destructive/20` but cannot drop a dark: class from the variant, so
 * in every dark tone the full-strength border overrode the softer one that the sign-in,
 * sign-up and reset-password error alerts ask for. The base `border-destructive/50` is the
 * token and follows every tone, so the dark: class is gone.
 */
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { Alert, AlertDescription } from '../alert'

afterEach(cleanup)

describe('Alert destructive border (P260-FX-05)', () => {
  it("keeps the caller's softer border, with no dark: border over it", () => {
    render(
      <Alert variant="destructive" className="border-destructive/20">
        <AlertDescription>Those details did not match.</AlertDescription>
      </Alert>,
    )
    const alert = screen.getByRole('alert')
    expect(alert.className).toContain('border-destructive/20')
    expect(alert.className).not.toContain('border-destructive/50')
    expect(alert.className).not.toMatch(/(^|\s)dark:border-/)
  })

  it('the plain destructive alert still draws its token border', () => {
    render(<Alert variant="destructive">Something went wrong.</Alert>)
    const alert = screen.getByRole('alert')
    expect(alert.className).toContain('border-destructive/50')
    expect(alert.className).toContain('text-destructive')
    expect(alert.className).not.toMatch(/(^|\s)dark:border-/)
  })
})
