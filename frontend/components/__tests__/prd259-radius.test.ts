/**
 * PRD-259 S-009 — `--radius` is defined globally.
 *
 * `tailwind.config.ts` maps rounded-lg/md/sm onto `var(--radius)`, but until
 * this story `app/globals.css` never defined the variable, so those utilities
 * drew square corners. Classic takes the shadcn default (0.5rem) in the first
 * `:root {` block; Studio takes 8px in the first `.studio {` block, and
 * `.studio.dark` inherits it. A skin overrides it (S-004a).
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const FRONTEND = path.resolve(__dirname, '..', '..')
const read = (rel: string) => readFileSync(path.join(FRONTEND, rel), 'utf8')
/** globals.css with its comments stripped — declarations only. */
const css = read('app/globals.css').replace(/\/\*[\s\S]*?\*\//g, '')

/** The text of the first block opened by `opener`, up to its matching brace. */
const blockOf = (opener: string): string => {
  const at = css.indexOf(opener)
  expect(at, `${opener} exists in globals.css`).toBeGreaterThan(-1)
  const open = at + opener.length - 1
  let depth = 0
  for (let i = open; i < css.length; i += 1) {
    if (css[i] === '{') depth += 1
    if (css[i] === '}') depth -= 1
    if (depth === 0) return css.slice(open + 1, i)
  }
  throw new Error(`${opener} is never closed`)
}

const RADIUS = /--radius\s*:\s*([^;]+);/g
const radiiIn = (text: string) => Array.from(text.matchAll(RADIUS), (m) => m[1].trim())

describe('PRD-259 S-009 — --radius is defined globally', () => {
  it('Classic: the first :root block sets the shadcn default 0.5rem', () => {
    expect(radiiIn(blockOf(':root {'))).toEqual(['0.5rem'])
  })

  it('Studio: the first .studio block sets 8px', () => {
    expect(radiiIn(blockOf('.studio {'))).toEqual(['8px'])
  })

  it('the dark blocks inherit it: .dark and .studio.dark set no radius', () => {
    expect(radiiIn(blockOf('.dark {'))).toEqual([])
    expect(radiiIn(blockOf('.studio.dark {'))).toEqual([])
  })

  it('those two are the only --radius declarations in the file', () => {
    expect(radiiIn(css)).toEqual(['0.5rem', '8px'])
  })

  it('tailwind.config.ts still reads it for rounded-lg/md/sm', () => {
    const tw = read('tailwind.config.ts')
    expect(tw).toContain("lg: 'var(--radius)'")
    expect(tw).toContain("md: 'calc(var(--radius) - 2px)'")
    expect(tw).toContain("sm: 'calc(var(--radius) - 4px)'")
  })
})
