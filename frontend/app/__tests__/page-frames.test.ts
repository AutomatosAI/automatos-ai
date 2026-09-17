/**
 * PRD-244 — every page wears one of the two frames.
 *
 * Gerard's review (2026-09-17): "SaaS has a lot more pages than local, so we
 * need to review the pages that were not done and convert — teams, workspace
 * admin etc." The shared primitives (`PageHeader`, `StatsBar`, `FilterTabs`)
 * render the Studio frame in the Studio style and the classic markup
 * otherwise, so a page built from them is a Studio page for free. A page that
 * hand-rolls its own `<h1>` block is not — it stays classic under Studio.
 *
 * This asserts every route reaches a frame: the shared `PageHeader`, or a
 * bespoke Studio editorial head (`cc-page` / `cc-h1`). A new page must do one
 * or the other, or be listed below with its reason.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync, readdirSync, statSync } from 'fs'
import path from 'path'

const APP = path.resolve(__dirname, '..')
const COMPONENTS = path.resolve(__dirname, '..', '..', 'components')

/** Routes that legitimately carry no page header. */
const NO_HEADER: Record<string, string> = {
  '/': 'redirects to the workspace entry point',
  '/chat': 'a conversation surface, not a document page',
  '/missions': 'redirects to the Assignments Missions tab',
  '/playbooks': 'redirects to the Assignments Playbooks tab',
  '/marketplace/widgets/[id]': 'item detail — the record’s own name is the title',
  '/accept-invitation': 'auth flow',
  '/reset-password': 'auth flow',
  '/sso-callback': 'auth flow',
  '/tools/callback': 'OAuth return, redirects',
  '/dev/reset-onboarding': 'developer utility',
  '/auth/signin/[[...rest]]': 'Clerk flow',
  '/auth/signup/[[...rest]]': 'Clerk flow',
  '/sign-in/[[...rest]]': 'Clerk flow',
  '/sign-up/[[...rest]]': 'Clerk flow',
}

const IMPORT = /import \{ ([A-Z][A-Za-z]+)[^}]*\} from ["']@\/components\/([^"']+)["']/g

function framed(file: string, depth = 0): boolean {
  const src = readFileSync(file, 'utf8')
  if (src.includes('<PageHeader') || src.includes('className="cc-page"') || src.includes('cc-h1')) return true
  if (depth > 1) return false
  for (const [, , rel] of src.matchAll(IMPORT)) {
    if (rel.includes('main-layout')) continue
    for (const cand of [path.join(COMPONENTS, `${rel}.tsx`), path.join(COMPONENTS, rel, 'index.tsx')]) {
      if (existsSync(cand) && framed(cand, depth + 1)) return true
    }
  }
  return false
}

function routes(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) {
      if (entry !== '__tests__') routes(full, out)
    } else if (entry === 'page.tsx') {
      const rel = path.relative(APP, dir)
      out.push(rel === '' ? '/' : `/${rel}`)
    }
  }
  return out
}

describe('every page wears a frame', () => {
  const all = routes(APP).sort()

  it('finds the app’s routes', () => {
    expect(all.length).toBeGreaterThan(20)
    expect(all).toContain('/team')
    expect(all).toContain('/admin/workspaces')
  })

  it('no route hand-rolls its page header', () => {
    const unframed = all.filter((r) => !(r in NO_HEADER) && !framed(path.join(APP, r === '/' ? '' : r, 'page.tsx')))
    expect(unframed, 'use PageHeader (or a cc-page editorial head), or record the route in NO_HEADER').toEqual([])
  })

  it('the admin surfaces Gerard named are on the shared frame', () => {
    for (const rel of ['admin/workspaces/page.tsx', 'admin/plugins/page.tsx', 'admin/plugins/upload/page.tsx', 'team/page.tsx']) {
      const src = readFileSync(path.join(APP, rel), 'utf8')
      expect(src.match(/<h1 className="text-(2|3)xl font-bold/), rel).toBeNull()
    }
    for (const rel of ['admin/workspaces/page.tsx', 'admin/plugins/page.tsx']) {
      expect(readFileSync(path.join(APP, rel), 'utf8'), rel).toContain('<StatsBar')
    }
  })
})
