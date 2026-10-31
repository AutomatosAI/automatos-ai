/**
 * PRD-265 W2 — Stripe's card fields in our theme. Stripe's fields are iframes, so they take
 * an `appearance` built from the theme tokens on <html> (HSL triplets such as "16 100% 50%",
 * turned into hex colours), with defaults where a token is unset (a server render, a test).
 */
import { afterEach, describe, expect, it } from 'vitest'
import { appearanceFrom, radiusOf, stripeAppearance, tokenToColor } from '@/lib/stripe-appearance'

const SET_ON_ROOT = ['--primary', '--background', '--foreground', '--radius']

afterEach(() => {
  SET_ON_ROOT.forEach((token) => document.documentElement.style.removeProperty(token))
})

describe('the theme tokens as Stripe colours', () => {
  it('turns an HSL triplet into a hex colour, and passes a colour literal through', () => {
    expect(tokenToColor('16 100% 50%')).toBe('#ff4400')
    expect(tokenToColor('0 0% 100%')).toBe('#ffffff')
    expect(tokenToColor('0 0% 0%')).toBe('#000000')
    expect(tokenToColor(' 30 12% 9% ')).toBe('#1a1714')
    expect(tokenToColor('#0a0d14')).toBe('#0a0d14')
    expect(tokenToColor('rgb(1, 2, 3)')).toBe('rgb(1, 2, 3)')
  })

  it('gives nothing for a value that is not a colour', () => {
    expect(tokenToColor('')).toBeNull()
    expect(tokenToColor('var(--x)')).toBeNull()
    expect(tokenToColor('url(javascript:alert(1))')).toBeNull()
  })

  it('reads the radius in px, from px or rem, with a default', () => {
    expect(radiusOf('0.5rem')).toBe('8px')
    expect(radiusOf('8px')).toBe('8px')
    expect(radiusOf('12px')).toBe('12px')
    expect(radiusOf('')).toBe('8px')
    expect(radiusOf('calc(1px + 2px)')).toBe('8px')
  })
})

describe('the appearance', () => {
  it('follows the theme: Studio light paints Stripe light, a dark ground paints it dark', () => {
    const studio: Record<string, string> = { '--primary': '30 12% 9%', '--background': '38 50% 92%', '--radius': '8px' }
    const light = appearanceFrom((token) => studio[token] ?? '')
    expect(light.theme).toBe('stripe')
    expect(light.labels).toBe('above')
    expect(light.variables?.colorPrimary).toBe('#1a1714')
    expect(light.variables?.borderRadius).toBe('8px')

    const darkGround: Record<string, string> = { '--background': '0 0% 6%' }
    const dark = appearanceFrom((token) => darkGround[token] ?? '')
    expect(dark.theme).toBe('night')
    expect(dark.variables?.colorBackground).toBe('#0f0f0f')
  })

  it('falls back to its defaults for every token the theme does not set', () => {
    const bare = appearanceFrom(() => '')
    expect(bare.theme).toBe('stripe')
    for (const value of [bare.variables?.colorPrimary, bare.variables?.colorText, bare.variables?.colorDanger]) {
      expect(value).toMatch(/^#[0-9a-f]{6}$/)
    }
  })

  it('reads the tokens the active theme sets on <html>', () => {
    document.documentElement.style.setProperty('--primary', '16 100% 50%')
    document.documentElement.style.setProperty('--background', '0 0% 6%')
    document.documentElement.style.setProperty('--radius', '0.75rem')

    const appearance = stripeAppearance()

    expect(appearance.variables?.colorPrimary).toBe('#ff4400')
    expect(appearance.theme).toBe('night')
    expect(appearance.variables?.borderRadius).toBe('12px')
  })
})
