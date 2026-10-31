/**
 * PRD-264 EUAI-11a — the colour half of the AI label's check (lib/ai-label/contrast.ts): computed colours
 * read as sRGB with alpha, translucent colours painted over their ground, and the WCAG contrast ratio.
 * A colour in another space goes through a canvas in a browser; jsdom has no OffscreenCanvas, so here it
 * reads as null, which the check treats as hidden.
 */
import { describe, expect, it } from 'vitest'

import { canvasGrounds, contrastRatio, isOpaque, over, parseCssColor } from '../contrast'

const WHITE = { r: 255, g: 255, b: 255, a: 1 }
const BLACK = { r: 0, g: 0, b: 0, a: 1 }

describe('parseCssColor', () => {
  it.each([
    ['rgb(12, 34, 56)', { r: 12, g: 34, b: 56, a: 1 }],
    ['rgba(12, 34, 56, 0.5)', { r: 12, g: 34, b: 56, a: 0.5 }],
    ['rgb(12 34 56 / 25%)', { r: 12, g: 34, b: 56, a: 0.25 }],
    ['color(srgb 1 0.5 0)', { r: 255, g: 127.5, b: 0, a: 1 }],
    ['color(srgb 0 0 0 / 0)', { r: 0, g: 0, b: 0, a: 0 }],
    ['transparent', { r: 0, g: 0, b: 0, a: 0 }],
    ['CanvasText', BLACK],
    ['Canvas', WHITE],
  ])('reads %s', (value, expected) => {
    expect(parseCssColor(value)).toEqual(expected)
  })

  it('reads nothing it cannot read as a colour (no canvas here): the check then treats the label as hidden', () => {
    expect(parseCssColor('')).toBeNull()
    expect(parseCssColor('oklch(0.7 0.1 200)')).toBeNull()
    expect(parseCssColor('rgb(nonsense)')).toBeNull()
  })
})

describe('contrast', () => {
  it('gives the WCAG ratios: 21 for black on white, 1 for a colour on itself', () => {
    expect(contrastRatio(BLACK, WHITE)).toBeCloseTo(21, 5)
    expect(contrastRatio(WHITE, BLACK)).toBeCloseTo(21, 5)
    expect(contrastRatio(WHITE, WHITE)).toBe(1)
  })

  it('paints a translucent colour over its ground', () => {
    expect(over({ r: 0, g: 0, b: 0, a: 0.5 }, WHITE)).toEqual({ r: 127.5, g: 127.5, b: 127.5, a: 1 })
    expect(over({ r: 0, g: 0, b: 0, a: 0 }, WHITE)).toEqual(WHITE)
    expect(isOpaque(WHITE)).toBe(true)
    expect(isOpaque({ ...WHITE, a: 0.5 })).toBe(false)
  })

  it("judges against white, and black too when the page may be dark", () => {
    expect(canvasGrounds(false)).toEqual([WHITE])
    expect(canvasGrounds(true)).toEqual([WHITE, BLACK])
  })
})
