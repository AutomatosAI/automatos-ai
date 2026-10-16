// @vitest-environment node
/**
 * P260-FX-14 (M31) — the first open on laptop heights. vitest runs with css: false and
 * jsdom lays nothing out, so the layout is proven from the stylesheets (the static reader of
 * lib/__tests__/css-cascade.ts): the hero grows and never shrinks below Auto and the
 * headline (no min-height: 0), the empty chat grows with the hero, the chips and the box,
 * and PageColumns' main column scrolls. On a short window the page then scrolls, instead
 * of "Hi, I'm Auto." overlapping the page header, the example chips and the message box.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { cssRules } from '@/lib/__tests__/css-cascade'

const PLAY = path.resolve(__dirname, '../..')
const FIRST_OPEN_CSS = path.join(PLAY, 'chat/first-open.module.css')
const COMPOSER_CSS = path.join(PLAY, 'chat/composer.module.css')
const COLUMNS_CSS = path.join(PLAY, 'shell/page-columns.module.css')
/** flex: grow into the free space, never shrink below the content (the approved app's layout). */
const GROWS_NEVER_SHRINKS = '1 0 auto'

/** The declarations of every rule that lists exactly `selector`, later ones winning, as the cascade does. */
function declarationsIn(css: string, selector: string): Map<string, string> {
  const found = new Map<string, string>()
  for (const rule of cssRules(css).filter((each) => each.selectors.includes(selector))) {
    for (const declaration of rule.body.split(';')) {
      const colon = declaration.indexOf(':')
      if (colon > 0) found.set(declaration.slice(0, colon).trim(), declaration.slice(colon + 1).trim().replace(/\s+/g, ' '))
    }
  }
  return found
}

function declarations(file: string, selector: string): Map<string, string> {
  return declarationsIn(fs.readFileSync(file, 'utf8'), selector)
}

describe('P260-FX-14 — the first open scrolls on a short window instead of overlapping (M31)', () => {
  it('reads the declarations of exactly that selector, the later rule winning', () => {
    const css = '.a { flex: 1; min-height: 0 } .a[data-x] .b { flex: none } .a, .c { flex:  1  0  auto }'
    expect(declarationsIn(css, '.a')).toEqual(new Map([['flex', GROWS_NEVER_SHRINKS], ['min-height', '0']]))
    expect(declarationsIn(css, '.a[data-x]').size).toBe(0)
  })

  it('lets the hero grow and never shrink: no min-height: 0, centred only while it fits', () => {
    const hero = declarations(FIRST_OPEN_CSS, '.hero')
    expect(hero.get('min-height')).not.toBe('0')
    expect(hero.get('flex')).toBe(GROWS_NEVER_SHRINKS)
    expect(hero.get('justify-content')).toBe('safe center')
  })

  it('lets the empty chat grow with its content, so the main column scrolls it', () => {
    expect(declarations(COMPOSER_CSS, '.chat[data-empty]').get('flex')).toBe(GROWS_NEVER_SHRINKS)
    expect(declarations(COLUMNS_CSS, '.main').get('overflow-y')).toBe('auto')
  })

  it('keeps a conversation scrolling its own list (the chat still shrinks to the column)', () => {
    expect(declarations(COMPOSER_CSS, '.chat').get('min-height')).toBe('0')
  })
})
