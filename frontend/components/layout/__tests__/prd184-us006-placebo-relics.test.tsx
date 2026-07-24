/**
 * PRD-184 US-006 — the frontend placebo relics are gone.
 *
 * Three dead / fabricating surfaces removed (no live nav or link targeted either
 * route — both were reachable only by typing the URL):
 *  - app/api-control/  — placebo control panel for the PRD-168-removed mock system.
 *  - app/styleguide/   — a styleguide page routed in prod but linked from nowhere.
 *  - the fabricated `workspaceMeta = 'pilot · 11 op'` default in studio-sidebar.tsx —
 *    a made-up "pilot · 11 op" literal with no truthful source. Dropped (not wired to
 *    a placeholder), mirroring the F038 fix that removed the fabricated `v0.11` string.
 *
 * Filesystem + source guards so the surfaces cannot silently regrow. Lockfiles
 * (PRD-209) and the /chat/[id] route (held S10) are intentionally untouched.
 */
import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

const FRONTEND = path.resolve(__dirname, '../../..')

// Canonical guard id (matched by scripts/ralph/acceptance-prd184.sh): test_no_placebo_routes
describe('PRD-184 US-006 · test_no_placebo_routes — placebo relics removed', () => {
  it('the /api-control placebo route is deleted', () => {
    expect(fs.existsSync(path.join(FRONTEND, 'app', 'api-control'))).toBe(false)
  })

  it('the /styleguide route is deleted', () => {
    expect(fs.existsSync(path.join(FRONTEND, 'app', 'styleguide'))).toBe(false)
  })

  it('the fabricated workspaceMeta pill is gone from the studio sidebar', () => {
    const src = fs.readFileSync(
      path.join(FRONTEND, 'components', 'layout', 'studio-sidebar.tsx'),
      'utf8',
    )
    expect(src).not.toContain('workspaceMeta')
    expect(src).not.toContain('pilot · 11 op')
    expect(src).not.toContain('sh-meta')
  })

  it('the held /chat/[id] route (S10) is untouched', () => {
    // Boundary proof: US-006 must NOT delete the live chat route.
    expect(fs.existsSync(path.join(FRONTEND, 'app', 'chat'))).toBe(true)
  })
})
