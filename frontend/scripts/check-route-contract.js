#!/usr/bin/env node
/**
 * PRD-182 S3 (F044) - frontend->backend route-contract check.
 *
 * The backend half of this contract already exists: `scripts/dump_routes.py`
 * emits `orchestrator/reports/route-manifest.json` and
 * `orchestrator/tests/test_route_manifest.py` proves it generates. PRD-155 S2
 * was supposed to land the FRONTEND half - asserting every path the UI calls
 * exists in that manifest - but it shipped with zero frontend files, so
 * "CI fails when the frontend calls a non-existent backend path" was untrue
 * (OS review F044). This script is that missing half.
 *
 * Scope: `frontend/lib/api-client.ts` - the single transport choke point. The
 * repo's eslint rule (`no-restricted-syntax`) bans raw `fetch('/api...')`, so
 * every backend call funnels through `apiClient`; auditing this one file
 * therefore covers the UI's real call surface without the false positives a
 * whole-repo string grep would produce (Next.js route file paths, cross-line
 * concatenations, non-call literals).
 *
 * Matching is structural, not literal: a frontend template segment `${expr}`
 * and a manifest path parameter `{param}` are the same "any value here" slot,
 * so both sides normalise every parameter segment to a single `*` token and a
 * query string is dropped. `/api/system/config/${key}` (frontend) and
 * `/api/system/config/{config_key}` (manifest) both become `/api/system/config/*`.
 *
 * Baselined, not aspirational (PRD-182 guiding rule): today's api-client already
 * calls paths absent from the manifest (dead methods or unmounted routers). Those
 * pre-existing drifts are recorded in `route-contract-baseline.json` beside this
 * script and are NOT failures. The check fails only on a NEW frontend call to a
 * path that is neither in the manifest nor in the baseline - the regression gate.
 * Shrink the baseline as those calls are fixed; never grow it silently.
 *
 * Exit 0 = no NEW contract break. Exit 1 = a new frontend call targets a path
 * absent from the manifest (and not baselined), or an input could not be read.
 *
 * Pure Node stdlib, no dependencies - runs identically in CI and locally.
 * Usage (from `frontend/`): node scripts/check-route-contract.js
 *   --update-baseline   rewrite the baseline to the current violation set
 */
'use strict'

const fs = require('fs')
const path = require('path')

const FRONTEND_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(FRONTEND_ROOT, '..')
const API_CLIENT = path.join(FRONTEND_ROOT, 'lib', 'api-client.ts')
const MANIFEST = path.join(
  REPO_ROOT,
  'orchestrator',
  'reports',
  'route-manifest.json'
)
const BASELINE = path.join(__dirname, 'route-contract-baseline.json')

// Path prefixes the backend actually serves (from the manifest's own shape).
// A frontend literal is only a route-contract candidate if it starts with one
// of these - this filters out client-only strings that happen to start with a
// slash (asset paths, next routes, etc.).
const BACKEND_PREFIXES = ['/api/', '/analytics/', '/health', '/metrics', '/ws/']

// ASCII SOH byte - a sentinel that can never appear in a URL path. A collapsed
// `${...}` interpolation becomes this so it survives the later `?`-split and
// still marks its whole path segment as a wildcard.
const INTERP_MARKER = String.fromCharCode(1)

/**
 * Normalise a path for structural comparison:
 *  - collapse every `${...}` interpolation (including ternary bodies with their
 *    own `?`/`:` like `${qs ? '?'+qs : ''}`) to a marker FIRST, so a query
 *    string baked into an interpolation doesn't confuse the `?`-split
 *  - drop any remaining query string / fragment (`?...`, `#...`)
 *  - collapse a trailing slash (except the root `/`)
 *  - replace every whole-segment parameter - a collapsed frontend interp or a
 *    manifest path parameter `{param}` - with a single `*` wildcard token
 */
function normalisePath(raw) {
  let p = raw
  let prev
  // Repeat until stable so a `${ ... {x} ... }` body collapses fully.
  do {
    prev = p
    p = p.replace(/\$\{[^{}]*\}/g, INTERP_MARKER)
  } while (p !== prev)
  p = p.split('?')[0].split('#')[0]
  if (p.length > 1 && p.endsWith('/')) p = p.slice(0, -1)
  const marker = INTERP_MARKER
  const segments = p.split('/').map((seg) => {
    // Manifest path parameter, e.g. `{config_key}` or `{model_id:path}`.
    if (seg.startsWith('{') && seg.endsWith('}')) return '*'
    if (seg.indexOf(marker) === -1) return seg
    // A segment that is ENTIRELY an interpolation is a path parameter -> wildcard.
    const stripped = seg.split(marker).join('')
    if (stripped === '') return '*'
    // A literal prefix followed by a trailing interpolation is the query-suffix
    // idiom `items${qs ? '?'+qs : ''}` - the interp expands to a query string or
    // '', never a new path segment. Keep the literal prefix, drop the suffix.
    if (seg.endsWith(marker) && !seg.startsWith(marker)) return stripped
    // Interpolation embedded mid-segment (rare) - cannot reason about it; wildcard.
    return '*'
  })
  return segments.join('/')
}

/**
 * Extract the endpoint literals passed to `this.request(...)` in api-client.ts.
 * Captures the FIRST argument when it is a single-quoted, double-quoted, or
 * template-literal string. A `this.request` whose first argument is an
 * identifier (the generic get/post wrappers forwarding `endpoint`, or a
 * locally-built `url` variable) is not a literal and is skipped - the literal it
 * ultimately carries appears verbatim elsewhere in the file and is captured
 * there.
 */
function extractFrontendPaths(source) {
  const found = new Set()
  // Anchor on each `this.request(` (optionally `<Type>`) and the opening quote
  // of the first argument, then hand off to a balanced scanner. A plain regex
  // capture breaks on nested template literals like
  // `/api/x${cond ? `?${q}` : ''}` (a backtick inside the `${...}`), so the
  // string body is read delimiter-aware instead.
  const anchor = /this\.request(?:<[^>]*>)?\(\s*(['"`])/g
  let m
  while ((m = anchor.exec(source)) !== null) {
    const quote = m[1]
    const start = anchor.lastIndex // first char inside the opening quote
    const literal = readStringLiteral(source, start, quote)
    if (literal !== null && literal.startsWith('/')) found.add(literal)
  }
  return found
}

/**
 * Read a JS string/template literal body starting just after its opening quote.
 * For a template literal (backtick) it tracks `${ ... }` interpolation depth and
 * nested backticks so an inner template does not prematurely end the outer one.
 * Returns the raw body (interpolations left intact for normalisePath to collapse)
 * or null if the literal is unterminated.
 */
function readStringLiteral(source, start, quote) {
  let i = start
  let depth = 0 // `${ }` nesting depth (template literals only)
  let body = ''
  while (i < source.length) {
    const ch = source[i]
    if (ch === '\\') {
      body += ch + (source[i + 1] || '')
      i += 2
      continue
    }
    if (quote === '`') {
      if (ch === '$' && source[i + 1] === '{') {
        depth++
        body += '${'
        i += 2
        continue
      }
      if (ch === '}' && depth > 0) {
        depth--
        body += '}'
        i++
        continue
      }
      // A backtick only closes the literal at interpolation depth 0.
      if (ch === '`' && depth === 0) return body
      body += ch
      i++
      continue
    }
    // Single/double quoted: no interpolation, closes on the matching quote.
    if (ch === quote) return body
    if (ch === '\n') return null // unterminated simple string
    body += ch
    i++
  }
  return null
}

function isBackendCandidate(p) {
  return BACKEND_PREFIXES.some((prefix) => p === prefix || p.startsWith(prefix))
}

function loadBaseline() {
  try {
    const data = JSON.parse(fs.readFileSync(BASELINE, 'utf8'))
    return new Set(Array.isArray(data.normalised) ? data.normalised : [])
  } catch (err) {
    // No baseline yet = every current violation is "new". The check will fail
    // and print the exact set to seed the baseline with --update-baseline.
    return new Set()
  }
}

function fail(msg) {
  console.error(msg)
  process.exit(1)
}

function main() {
  const updateBaseline = process.argv.includes('--update-baseline')

  let source
  try {
    source = fs.readFileSync(API_CLIENT, 'utf8')
  } catch (err) {
    fail(`route-contract: cannot read api-client at ${API_CLIENT}: ${err.message}`)
  }

  let manifest
  try {
    manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'))
  } catch (err) {
    fail(
      `route-contract: cannot read route manifest at ${MANIFEST}: ${err.message}\n` +
        `Regenerate it with:  (cd orchestrator && python3 -m scripts.dump_routes)`
    )
  }

  const manifestRoutes = Array.isArray(manifest.routes) ? manifest.routes : []
  if (manifestRoutes.length === 0) {
    fail('route-contract: manifest has zero routes - refusing to pass vacuously')
  }

  const manifestPaths = new Set()
  for (const r of manifestRoutes) {
    if (r && typeof r.path === 'string') manifestPaths.add(normalisePath(r.path))
  }

  const rawPaths = extractFrontendPaths(source)
  const candidates = [...rawPaths].filter(isBackendCandidate)
  if (candidates.length === 0) {
    fail(
      'route-contract: extracted zero backend path literals from api-client.ts - ' +
        'the extractor is broken or the transport moved. Refusing to pass vacuously.'
    )
  }

  // All current violations (normalised), with a raw example for each.
  const currentViolations = new Map() // normalised -> raw example
  for (const raw of candidates.sort()) {
    const norm = normalisePath(raw)
    if (!manifestPaths.has(norm) && !currentViolations.has(norm)) {
      currentViolations.set(norm, raw)
    }
  }

  if (updateBaseline) {
    const normalised = [...currentViolations.keys()].sort()
    const payload = {
      _comment:
        'PRD-182 S3 (F044) route-contract baseline. Pre-existing api-client ' +
        'calls to paths not in orchestrator/reports/route-manifest.json. The ' +
        'check fails only on a NEW violation not listed here. Shrink this list ' +
        'as calls are fixed or routers mounted; never grow it silently. ' +
        'Regenerate with: node scripts/check-route-contract.js --update-baseline',
      normalised,
      examples: Object.fromEntries(
        normalised.map((n) => [n, currentViolations.get(n)])
      ),
    }
    fs.writeFileSync(BASELINE, JSON.stringify(payload, null, 2) + '\n')
    console.log(
      `route-contract: wrote baseline with ${normalised.length} pre-existing ` +
        `violation(s) to ${path.relative(REPO_ROOT, BASELINE)}`
    )
    process.exit(0)
  }

  const baseline = loadBaseline()
  const newViolations = [...currentViolations.entries()].filter(
    ([norm]) => !baseline.has(norm)
  )

  console.log(
    `route-contract: ${candidates.length} frontend call path(s) vs ` +
      `${manifestPaths.size} manifest path(s); ` +
      `${currentViolations.size} pre-existing violation(s) baselined, ` +
      `${newViolations.length} new.`
  )

  if (newViolations.length > 0) {
    console.error(
      `\n[FAIL] ${newViolations.length} NEW frontend call(s) target a backend ` +
        `path NOT in route-manifest.json and NOT baselined:\n`
    )
    for (const [norm, raw] of newViolations.sort()) {
      console.error(`   ${raw}   (normalised: ${norm})`)
    }
    console.error(
      '\nEither the route was renamed/removed on the backend, the manifest is ' +
        'stale, or this is a genuinely non-existent path.\n' +
        ' - regenerate the manifest:  (cd orchestrator && python3 -m scripts.dump_routes)\n' +
        ' - if intentional and pre-existing, re-baseline: node scripts/check-route-contract.js --update-baseline\n'
    )
    process.exit(1)
  }

  console.log('[OK] no new frontend->backend route-contract violation.')
  process.exit(0)
}

main()
