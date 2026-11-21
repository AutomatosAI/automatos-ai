// @vitest-environment node
/**
 * The CI job "Prod images built with default args carry production behaviour"
 * (.github/workflows/test.yml) builds the hosted frontend image and greps the compiled output for
 * `"local"==="saas".trim()` (and for no `"local"==="local".trim()`). That text exists only because
 * `resolveEdition` in lib/edition-name.ts reads the variable into `raw` once and compares it with
 * `'local'` once: the minifier then inlines `raw` (inlined at build as "saas") into the comparison.
 * Used twice, `raw` stays a variable and the text the job reads is gone, so the job fails.
 *
 * This reads the SOURCE and fails as soon as that comparison stops having that shape.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

const SOURCE = readFileSync(path.resolve(__dirname, '..', 'edition-name.ts'), 'utf8')
const READ = "const raw = (process.env.NEXT_PUBLIC_AUTH_EDITION || '').trim().toLowerCase()"
const COMPARE = "return raw === 'local' ? 'local' : 'saas'"

function bodyOf(name: string): string {
  const start = SOURCE.indexOf(`export function ${name}(`)
  expect(start, `${name} is gone from lib/edition-name.ts`).toBeGreaterThan(-1)
  const open = SOURCE.indexOf('{', start)
  const close = SOURCE.indexOf('\n}', open)
  return SOURCE.slice(open + 1, close)
}

describe("resolveEdition's 'local' comparison keeps the shape the prod image check reads", () => {
  it('reads the variable into raw once and compares it with local once, as on main', () => {
    const body = bodyOf('resolveEdition')
    expect(body).toContain(READ)
    expect(body).toContain(COMPARE)
    expect(body.indexOf(READ)).toBeLessThan(body.indexOf(COMPARE))
    expect(body.match(/\braw\b/g)).toHaveLength(2)
  })

  it("compares with 'local' nowhere else", () => {
    expect(SOURCE.split("=== 'local'")).toHaveLength(2)
  })
})
