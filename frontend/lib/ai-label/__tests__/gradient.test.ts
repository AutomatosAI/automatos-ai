/**
 * PRD-264 EUAI-11a — the AI label's ground check reads a gradient as the browser paints it
 * (lib/ai-label/gradient-parse.ts, gradient-paint.ts): the one colour at a point where the geometry reads
 * (a linear gradient's angle or corner, a radial gradient's size and centre, a conic gradient's start, stop
 * positions in px, percentages and calc(), hints, repeating periods), and every colour it could paint where
 * the geometry doesn't. A layer that is no gradient reads as null, and a stop it can't read as 'unreadable':
 * the check fails closed on both. jsdom has no canvas, so a named colour here reads as unreadable.
 */
import { describe, expect, it } from 'vitest'

import type { Rgba } from '../contrast'
import { cssLength, splitTopLevel } from '../css-tokens'
import { type Gradient, parseGradient } from '../gradient-parse'
import { BETWEEN_STOPS, gradientColours, gradientColoursAt } from '../gradient-paint'

const BLACK: Rgba = { r: 0, g: 0, b: 0, a: 1 }
const WHITE: Rgba = { r: 255, g: 255, b: 255, a: 1 }
const RED: Rgba = { r: 255, g: 0, b: 0, a: 1 }
const BLUE: Rgba = { r: 0, g: 0, b: 255, a: 1 }
const GREY: Rgba = { r: 127.5, g: 127.5, b: 127.5, a: 1 }
const CLEAR: Rgba = { r: 0, g: 0, b: 0, a: 0 }

function gradient(text: string, currentColour = 'rgb(0, 0, 0)'): Gradient {
  const parsed = parseGradient(text, currentColour)
  if (parsed === null || parsed === 'unreadable') throw new Error(`not read as a gradient: ${text}`)
  return parsed
}

function expectColours(actual: readonly Rgba[], expected: readonly Rgba[]): void {
  expect(actual).toHaveLength(expected.length)
  actual.forEach((colour, index) => {
    for (const channel of ['r', 'g', 'b', 'a'] as const) expect(colour[channel]).toBeCloseTo(expected[index][channel], 4)
  })
}

describe('reading CSS values (css-tokens.ts)', () => {
  it('splits lists outside parentheses and quotes, and reads px, percentages and calc()', () => {
    expect(splitTopLevel('rgb(1, 2, 3) 10px, url("a,b"), none', ',')).toEqual(['rgb(1, 2, 3) 10px', 'url("a,b")', 'none'])
    expect(splitTopLevel('rgba(0, 0, 0, 0.5)  0px 1px', ' ')).toEqual(['rgba(0, 0, 0, 0.5)', '0px', '1px'])
    expect(cssLength('12.5px')).toEqual({ px: 12.5, pct: 0 })
    expect(cssLength('40%')).toEqual({ px: 0, pct: 40 })
    expect(cssLength('0')).toEqual({ px: 0, pct: 0 })
    expect(cssLength('calc(100% - 7px)')).toEqual({ px: -7, pct: 100 })
    expect(cssLength('calc(2em * 3)')).toBeNull()
    expect(cssLength('auto')).toBeNull()
  })
})

describe('parseGradient (PRD-264 EUAI-11a)', () => {
  it('reads no gradient from an image that is none (url(), image-set()): the ground check fails closed on it', () => {
    expect(parseGradient('url("ground.png")', 'rgb(0, 0, 0)')).toBeNull()
    expect(parseGradient('image-set("a.png" 1x)', 'rgb(0, 0, 0)')).toBeNull()
  })

  it("reads a stop it can't read as unreadable (no canvas here for a named colour), so the ground fails closed", () => {
    expect(parseGradient('linear-gradient(red, rgb(0, 0, 255))', 'rgb(0, 0, 0)')).toBe('unreadable')
  })

  it("reads currentcolor as the element's own colour, and two positions as two stops", () => {
    const read = gradient('linear-gradient(currentcolor 0px 1px, transparent 1px)', 'rgb(10, 20, 30)')
    expect(read.entries).toEqual([
      { colour: { r: 10, g: 20, b: 30, a: 1 }, at: { px: 0, pct: 0 } },
      { colour: { r: 10, g: 20, b: 30, a: 1 }, at: { px: 1, pct: 0 } },
      { colour: CLEAR, at: { px: 1, pct: 0 } },
    ])
  })

  it('keeps the stops but no geometry for a colour space or a prefixed gradient it does not read', () => {
    expect(gradient('linear-gradient(in oklch, rgb(0, 0, 0), rgb(255, 255, 255))').geometry).toBeNull()
    expect(gradient('-webkit-linear-gradient(left, rgb(0, 0, 0), rgb(255, 255, 255))').geometry).toBeNull()
    expect(gradient('linear-gradient(in srgb, rgb(0, 0, 0), rgb(255, 255, 255))').geometry).toEqual({ kind: 'linear', direction: { angle: 180 } })
  })
})

describe('gradientColoursAt (PRD-264 EUAI-11a)', () => {
  it('reads a linear gradient at the point: along an angle, a side, and to a corner', () => {
    const box = { width: 100, height: 10 }
    expectColours(gradientColoursAt(gradient('linear-gradient(to right, rgb(0, 0, 0), rgb(255, 255, 255))'), box, 50, 5), [GREY])
    expectColours(gradientColoursAt(gradient('linear-gradient(90deg, rgb(0, 0, 0), rgb(255, 255, 255))'), box, 0, 5), [BLACK])
    expectColours(gradientColoursAt(gradient('linear-gradient(rgb(0, 0, 0), rgb(255, 255, 255))'), box, 30, 10), [WHITE])
    // to a corner, the gradient's half-way line runs through the other two corners
    const corner = gradient('linear-gradient(to top right, rgb(255, 0, 0), rgb(0, 0, 255))')
    const wide = { width: 200, height: 100 }
    expectColours(gradientColoursAt(corner, wide, 0, 100), [RED])
    expectColours(gradientColoursAt(corner, wide, 200, 0), [BLUE])
    expectColours(gradientColoursAt(corner, wide, 0, 0), [{ r: 127.5, g: 0, b: 127.5, a: 1 }])
  })

  it("reads a radial glow as clear past its ending shape: HUD's backdrop glow at the label's corner", () => {
    const glow = gradient('radial-gradient(640px 520px at 50% 46%, rgba(227, 236, 240, 0.2), rgba(0, 0, 0, 0) 70%)')
    const viewport = { width: 1440, height: 900 }
    expectColours(gradientColoursAt(glow, viewport, 1417, 880), [CLEAR])
    expectColours(gradientColoursAt(glow, viewport, 720, 414), [{ r: 227, g: 236, b: 240, a: 0.2 }])
  })

  it('sizes a radial gradient by its extent keyword', () => {
    const halfway = gradient('radial-gradient(closest-side, rgb(0, 0, 0), rgb(255, 255, 255))')
    expectColours(gradientColoursAt(halfway, { width: 100, height: 50 }, 75, 25), [GREY])
    expectColours(gradientColoursAt(halfway, { width: 100, height: 50 }, 50, 37.5), [GREY])
  })

  it('folds a repeating gradient into its period, and places hard stops in px', () => {
    const scan = gradient('repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.18) 0px, rgba(0, 0, 0, 0.18) 1px, rgba(0, 0, 0, 0) 1px, rgba(0, 0, 0, 0) 3px)')
    const box = { width: 10, height: 30 }
    expectColours(gradientColoursAt(scan, box, 5, 29.5), [{ r: 0, g: 0, b: 0, a: 0.18 }])
    expectColours(gradientColoursAt(scan, box, 5, 28), [CLEAR])
    expectColours(gradientColoursAt(scan, box, 5, 26.5), [{ r: 0, g: 0, b: 0, a: 0.18 }])
    const edge = gradient('linear-gradient(to right, rgb(0, 0, 0) calc(100% - 10px), rgb(255, 255, 255) calc(100% - 10px))')
    expectColours(gradientColoursAt(edge, { width: 100, height: 10 }, 85, 5), [BLACK])
    expectColours(gradientColoursAt(edge, { width: 100, height: 10 }, 95, 5), [WHITE])
  })

  it("bends the blend through a hint, and turns a conic gradient from its start", () => {
    const hinted = gradient('linear-gradient(to right, rgb(0, 0, 0), 25%, rgb(255, 255, 255))')
    expectColours(gradientColoursAt(hinted, { width: 100, height: 10 }, 25, 5), [GREY])
    const turned = gradient('conic-gradient(from 90deg, rgb(255, 0, 0), rgb(0, 0, 255))')
    expectColours(gradientColoursAt(turned, { width: 100, height: 100 }, 90, 50), [RED])
    expectColours(gradientColoursAt(turned, { width: 100, height: 100 }, 50, 10), [{ r: 63.75, g: 0, b: 191.25, a: 1 }])
  })

  it('takes every colour the gradient could paint where its geometry does not read: its stops and the blends between', () => {
    const unread = gradient('linear-gradient(in oklch, rgb(0, 0, 0), rgb(255, 255, 255))')
    const every = gradientColours(unread)
    expect(every).toHaveLength(2 + BETWEEN_STOPS.length)
    expectColours(gradientColoursAt(unread, { width: 100, height: 10 }, 50, 5), every)
    expect(every.some((colour) => Math.round(colour.r) === 128 || Math.round(colour.r) === 127)).toBe(true)
    expectColours(gradientColoursAt(gradient('linear-gradient(rgb(0, 0, 0), rgb(255, 255, 255))'), { width: 0, height: 10 }, 0, 5), every)
  })
})
