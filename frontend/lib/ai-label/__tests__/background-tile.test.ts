/**
 * PRD-264 EUAI-11a — where a background layer paints under a point, for the AI label's ground check
 * (lib/ai-label/background-tile.ts): its clip, origin, size, position, repeat and attachment, so a small
 * no-repeat gradient (a HUD panel's corner bracket) counts only where it is, a grid only on its lines, and a
 * value the check doesn't read leaves the layer 'maybe' there. jsdom computes none of these longhands, so the
 * computed style here is a plain stand-in.
 */
import { describe, expect, it } from 'vitest'

import { colourReaches, tileAt } from '../background-tile'

const RECT = { left: 100, top: 100, width: 200, height: 100 }
const VIEW = { innerWidth: 1440, innerHeight: 900 } as Window

function styleOf(values: Readonly<Record<string, string>>): CSSStyleDeclaration {
  return { getPropertyValue: (name: string) => values[name] ?? '' } as unknown as CSSStyleDeclaration
}

describe('tileAt (PRD-264 EUAI-11a)', () => {
  it("covers the whole box by default, and says where in it the point is", () => {
    expect(tileAt(styleOf({}), 0, RECT, VIEW, { x: 150, y: 120 })).toEqual({ covers: 'yes', width: 200, height: 100, x: 50, y: 20 })
  })

  it("lays a small no-repeat tile where its position puts it: a corner bracket isn't under the middle of the box", () => {
    const bracket = styleOf({
      'background-size': '14px 1.5px',
      'background-repeat': 'no-repeat',
      'background-position-x': '100%',
      'background-position-y': '100%',
    })
    expect(tileAt(bracket, 0, RECT, VIEW, { x: 150, y: 150 })).toEqual({ covers: 'no' })
    expect(tileAt(bracket, 0, RECT, VIEW, { x: 295, y: 199 })).toEqual({ covers: 'yes', width: 14, height: 1.5, x: 9, y: 0.5 })
  })

  it('reads an edge offset the way Firefox reports it, and calc() the way Chromium does', () => {
    const tile = { 'background-size': '12px 30px', 'background-repeat': 'no-repeat' }
    const edges = styleOf({ ...tile, 'background-position-x': 'right 7px', 'background-position-y': 'bottom 7px' })
    const calc = styleOf({ ...tile, 'background-position-x': 'calc(100% - 7px)', 'background-position-y': 'calc(100% - 7px)' })
    for (const style of [edges, calc]) {
      expect(tileAt(style, 0, RECT, VIEW, { x: 285, y: 170 })).toEqual({ covers: 'yes', width: 12, height: 30, x: 4, y: 7 })
      expect(tileAt(style, 0, RECT, VIEW, { x: 150, y: 170 })).toEqual({ covers: 'no' })
    }
  })

  it('repeats a tile across the box, reads each layer from its list, and lays a fixed layer on the viewport', () => {
    const grid = styleOf({ 'background-size': '40px 40px, auto', 'background-repeat': 'repeat, no-repeat' })
    expect(tileAt(grid, 0, RECT, VIEW, { x: 145, y: 182 })).toEqual({ covers: 'yes', width: 40, height: 40, x: 5, y: 2 })
    expect(tileAt(grid, 1, RECT, VIEW, { x: 145, y: 182 })).toEqual({ covers: 'yes', width: 200, height: 100, x: 45, y: 82 })
    const fixed = styleOf({ 'background-attachment': 'fixed' })
    expect(tileAt(fixed, 0, RECT, VIEW, { x: 150, y: 120 })).toEqual({ covers: 'yes', width: 1440, height: 900, x: 150, y: 120 })
  })

  it('paints nothing outside its clip box or for a layer clipped to text, and says maybe where it reads no layout', () => {
    const bordered = { 'border-top-width': '10px', 'border-right-width': '10px', 'border-bottom-width': '10px', 'border-left-width': '10px' }
    expect(tileAt(styleOf({ ...bordered, 'background-clip': 'padding-box' }), 0, RECT, VIEW, { x: 105, y: 150 })).toEqual({ covers: 'no' })
    expect(tileAt(styleOf({ 'background-clip': 'text' }), 0, RECT, VIEW, { x: 150, y: 150 })).toEqual({ covers: 'no' })
    expect(tileAt(styleOf({ 'background-size': 'min(10px, 5%)' }), 0, RECT, VIEW, { x: 150, y: 150 })).toEqual({ covers: 'maybe' })
    expect(tileAt(styleOf({ 'background-repeat': 'space' }), 0, RECT, VIEW, { x: 150, y: 150 })).toEqual({ covers: 'maybe' })
  })
})

describe('colourReaches (PRD-264 EUAI-11a)', () => {
  it("reaches the point inside the bottom layer's clip box, and never for a colour clipped to text", () => {
    const bordered = { 'border-top-width': '10px', 'border-right-width': '10px', 'border-bottom-width': '10px', 'border-left-width': '10px' }
    expect(colourReaches(styleOf(bordered), RECT, { x: 105, y: 150 })).toBe(true)
    expect(colourReaches(styleOf({ ...bordered, 'background-clip': 'padding-box' }), RECT, { x: 105, y: 150 })).toBe(false)
    expect(colourReaches(styleOf({ 'background-clip': 'text' }), RECT, { x: 150, y: 150 })).toBe(false)
  })
})
