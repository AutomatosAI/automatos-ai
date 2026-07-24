#!/usr/bin/env node
/**
 * PRD-182 S1 (F034) - baselined tsc gate for the frontend CI lane.
 *
 * The frontend's 148k lines ship with `ignoreBuildErrors:true` in
 * next.config.js, so `next build` never type-checks and ~hundreds of
 * pre-existing TS errors go unenforced (OS review F034). Flipping that flag to
 * false would break every deploy on day one, so it stays. This gate is the
 * additive alternative: it runs `tsc --noEmit` as a SEPARATE CI step and fails
 * only when the error count REGRESSES above a recorded baseline - a ratchet, not
 * an aspirational zero.
 *
 * The baseline is the honest, MEASURED floor recorded in `.tsc-baseline.json`.
 * A new type error pushes the count over it and fails the job; fixing errors
 * lets you ratchet the baseline down with --update. The count only ever moves
 * in the safe direction automatically (fewer errors always passes).
 *
 * Usage (from `frontend/`):
 *   node scripts/tsc-baseline-check.js            # gate: fail if count > baseline
 *   node scripts/tsc-baseline-check.js --update   # re-record the baseline to now
 *
 * Pure Node stdlib. Invokes the local `tsc` (node_modules/.bin/tsc) via npx-free
 * resolution so it runs identically in CI and locally.
 */
'use strict'

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const FRONTEND_ROOT = path.resolve(__dirname, '..')
const BASELINE_FILE = path.join(FRONTEND_ROOT, '.tsc-baseline.json')

/** Run `tsc --noEmit` and return { count, output }. */
function runTsc() {
  // Resolve the project-local tsc binary; fall back to PATH.
  const localTsc = path.join(
    FRONTEND_ROOT,
    'node_modules',
    '.bin',
    process.platform === 'win32' ? 'tsc.cmd' : 'tsc'
  )
  const bin = fs.existsSync(localTsc) ? localTsc : 'tsc'
  const res = spawnSync(bin, ['--noEmit'], {
    cwd: FRONTEND_ROOT,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  })
  const output = `${res.stdout || ''}${res.stderr || ''}`
  if (res.error) {
    console.error(`tsc-baseline: failed to run tsc: ${res.error.message}`)
    process.exit(1)
  }
  // Count diagnostic lines of the form `path(line,col): error TSxxxx: ...`.
  const matches = output.match(/error TS[0-9]+/g)
  const count = matches ? matches.length : 0
  return { count, output }
}

function readBaseline() {
  try {
    const data = JSON.parse(fs.readFileSync(BASELINE_FILE, 'utf8'))
    if (typeof data.maxErrors === 'number') return data.maxErrors
  } catch (err) {
    // handled by caller
  }
  return null
}

function main() {
  const update = process.argv.includes('--update')
  const { count, output } = runTsc()

  if (update) {
    const payload = {
      _comment:
        'PRD-182 S1 (F034) tsc error baseline. MEASURED floor of pre-existing ' +
        'TypeScript errors under a strict tsconfig. CI runs `tsc --noEmit` and ' +
        'fails only when the count EXCEEDS maxErrors (a regression). Ratchet this ' +
        'number DOWN as errors are fixed; never raise it to silence a regression. ' +
        'Regenerate with: node scripts/tsc-baseline-check.js --update',
      maxErrors: count,
      measuredOn:
        'node ' + process.version + ' (CI runs node 20; fewer-errors always passes)',
    }
    fs.writeFileSync(BASELINE_FILE, JSON.stringify(payload, null, 2) + '\n')
    console.log(`tsc-baseline: recorded maxErrors=${count} to ${path.relative(FRONTEND_ROOT, BASELINE_FILE)}`)
    process.exit(0)
  }

  const baseline = readBaseline()
  if (baseline === null) {
    console.error(
      `tsc-baseline: no baseline at ${BASELINE_FILE}. Seed it with:\n` +
        `   node scripts/tsc-baseline-check.js --update`
    )
    process.exit(1)
  }

  console.log(`tsc-baseline: ${count} TypeScript error(s); baseline maxErrors=${baseline}.`)

  if (count > baseline) {
    console.error(
      `\n[FAIL] TypeScript errors regressed: ${count} > baseline ${baseline} ` +
        `(+${count - baseline}).\n` +
        `Fix the new type error(s) below. Do NOT raise the baseline to hide them.\n`
    )
    // Print the diagnostics so the regression is actionable in the CI log.
    const lines = output.split('\n').filter((l) => /error TS[0-9]+/.test(l))
    for (const l of lines) console.error('   ' + l)
    process.exit(1)
  }

  if (count < baseline) {
    console.log(
      `[OK] ${baseline - count} fewer error(s) than baseline. ` +
        `Consider ratcheting the baseline down: node scripts/tsc-baseline-check.js --update`
    )
  } else {
    console.log('[OK] TypeScript error count is at the baseline (no regression).')
  }
  process.exit(0)
}

main()
