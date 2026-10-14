// @vitest-environment node
/**
 * PRD-259 S-010a — `dark:` classes generate rules.
 *
 * Tailwind 3.3.3's darkVariants (node_modules/tailwindcss/lib/corePlugins.js)
 * knows only 'class' and 'media'; the ['variant', …] form arrived in 3.4.1, so
 * the config's old `darkMode: ['variant', ['&:is(.dark *)']]` added no dark
 * variant and every `dark:` class was dead. ['class', '.dark'] gives the same
 * `:is(.dark &)` selector. This compiles a probe against the real config.
 */
import { describe, it, expect } from 'vitest'
import postcss from 'postcss'
import tailwindcss from 'tailwindcss'
import config from '../../tailwind.config'

const PROBE = '<div class="dark:bg-background"></div>'
const DARK_RULE = ':is(.dark .dark\\:bg-background)'

describe('PRD-259 S-010a — dark: works under Tailwind 3.3.3', () => {
  it("the config's darkMode is the class form keyed on .dark", () => {
    expect(config.darkMode).toEqual(['class', '.dark'])
  })

  it('a dark: class compiles to a rule scoped under .dark', async () => {
    const plugin = tailwindcss({ ...config, content: [{ raw: PROBE, extension: 'html' }] })
    const result = await postcss([plugin]).process('@tailwind utilities;', { from: undefined })
    expect(result.css).toContain(DARK_RULE)
  })
})
