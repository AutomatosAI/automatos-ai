#!/usr/bin/env node
/**
 * PRD-262 US-049 - public/ holds only what the app serves, at the size it's shown.
 *
 * Everything under frontend/public ships in the frontend image as its own layer, and
 * every browser downloads the logos it shows. The logos are shown at 24-64 px
 * (components/ui/tool-logo.tsx, lib/logo-service.ts), so they are kept at 128 px or less.
 * Three of them were not images at all (two HTML pages and a file in no image format),
 * and 73 were ICO files named .png. This check keeps it that way. It fails when:
 *
 *   - the files under public/ add up to more than the budget in public-budget.json,
 *     beside this script. Raise the budget only in the pull request that needs it,
 *     and say why in its reason;
 *   - a file under public/logos is not what its name says by its magic bytes: a .png
 *     must be a PNG and a .svg an SVG, and no other kind of file belongs there;
 *   - a logo PNG is wider or taller than 256 px.
 *
 * The files counted are the ones git tracks or would track: generated files that the
 * frontend's .gitignore leaves out (public/theme-sdk, written by the build) are not.
 *
 * Exit 0 = within budget and every logo is a real PNG or SVG. Exit 1 = a problem, or an
 * input could not be read.
 *
 * Pure Node stdlib, no dependencies - runs identically in CI and locally.
 * Usage (from `frontend/`): node scripts/check-public-assets.js
 */
'use strict'

const fs = require('fs')
const path = require('path')
const { execFileSync } = require('child_process')

const FRONTEND = path.resolve(__dirname, '..')
const BUDGET_FILE = path.join(__dirname, 'public-budget.json')
const PUBLIC_DIR = 'public'
const LOGOS_DIR = 'public/logos/'
const MAX_LOGO_SIDE = 256
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
// IHDR is the first chunk: 8-byte signature, 4-byte length, 'IHDR', then width and height.
const IHDR_TYPE_OFFSET = 12
const IHDR_WIDTH_OFFSET = 16
const IHDR_HEIGHT_OFFSET = 20
const PNG_MIN_BYTES = 24
const SVG_SNIFF_BYTES = 1024
const HEAD_HEX_BYTES = 8
// What may come before an SVG's root element: whitespace, an XML declaration, comments
// and an svg doctype, whose internal subset (Illustrator exports have one) may hold '>'.
// Every repeat has one way to match (CodeQL js/redos): one whitespace character per
// round, not \s+, and in the doctype a plain character is neither bracket while the
// internal subset runs to its first ']'.
const SVG_PROLOG = /^(?:\s|<\?xml[^>]*\?>|<!--[\s\S]*?-->|<!DOCTYPE\s+svg(?:[^>[\]]|\[[^\]]*\])*>)*/i
const SVG_ROOT = /^<svg[\s>]/
const KINDS = { '.png': 'png', '.svg': 'svg' }
const ARTICLE = { png: 'a PNG', svg: 'an SVG' }

/** 'png', 'svg' or null: what the bytes are, whatever the file is called. */
function sniff(buf) {
  if (buf.length >= PNG_MIN_BYTES && buf.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    return buf.toString('latin1', IHDR_TYPE_OFFSET, IHDR_TYPE_OFFSET + 4) === 'IHDR' ? 'png' : null
  }
  const head = buf.subarray(0, SVG_SNIFF_BYTES).toString('utf8').replace(/^\uFEFF/, '')
  return SVG_ROOT.test(head.replace(SVG_PROLOG, '')) ? 'svg' : null
}

/** A PNG's width and height, read from its IHDR chunk. */
function pngSize(buf) {
  return { width: buf.readUInt32BE(IHDR_WIDTH_OFFSET), height: buf.readUInt32BE(IHDR_HEIGHT_OFFSET) }
}

/** The problems with one file under public/logos: its name relative to frontend/, and its bytes. */
function logoProblems(rel, buf) {
  const expected = KINDS[path.extname(rel).toLowerCase()]
  if (!expected) return [`${rel}: only .png and .svg files belong in ${LOGOS_DIR}`]
  const kind = sniff(buf)
  if (kind !== expected) {
    const head = buf.subarray(0, HEAD_HEX_BYTES).toString('hex').replace(/(..)(?!$)/g, '$1 ')
    const found = kind ? ARTICLE[kind] : `neither a PNG nor an SVG (it starts ${head || 'with nothing'})`
    return [`${rel}: named ${path.extname(rel)}, but by its magic bytes it is ${found}`]
  }
  if (kind !== 'png') return []
  const { width, height } = pngSize(buf)
  if (width <= MAX_LOGO_SIDE && height <= MAX_LOGO_SIDE) return []
  return [`${rel}: ${width}x${height} px, over the ${MAX_LOGO_SIDE} px limit (logos are shown at 64 px or less)`]
}

/** The files under public/ that git tracks or would track, relative to frontend/. */
function listPublicFiles(frontendDir) {
  const out = execFileSync(
    'git',
    ['ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', PUBLIC_DIR],
    { cwd: frontendDir, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
  )
  return [...new Set(out.split('\0').filter(Boolean))].sort()
}

/** Bytes per file; a tracked file deleted from the working tree is skipped. */
function fileSizes(frontendDir, files) {
  return files.flatMap((rel) => {
    const stat = fs.statSync(path.join(frontendDir, rel), { throwIfNoEntry: false })
    return stat && stat.isFile() ? [{ rel, bytes: stat.size }] : []
  })
}

/** Bytes per top-level entry under public/, largest first: what to look at when over budget. */
function breakdown(sizes) {
  const byTop = new Map()
  for (const { rel, bytes } of sizes) {
    const top = rel.split('/').slice(0, 2).join('/')
    byTop.set(top, (byTop.get(top) || 0) + bytes)
  }
  return [...byTop.entries()].sort((a, b) => b[1] - a[1])
}

/** The budget file, checked: a positive whole number of bytes and a reason. */
function readBudget(file = BUDGET_FILE) {
  const budget = JSON.parse(fs.readFileSync(file, 'utf8'))
  if (!Number.isInteger(budget.max_bytes) || budget.max_bytes <= 0) {
    throw new Error(`${file}: max_bytes must be a positive whole number of bytes`)
  }
  if (typeof budget.reason !== 'string' || !budget.reason.trim()) {
    throw new Error(`${file}: every budget needs a reason`)
  }
  return budget
}

/** Every problem with public/: the budget, then each logo. */
function checkPublic(frontendDir, files, budget) {
  const sizes = fileSizes(frontendDir, files)
  const total = sizes.reduce((sum, { bytes }) => sum + bytes, 0)
  const problems = total > budget.max_bytes
    ? [`public/ is ${total} bytes, over its budget of ${budget.max_bytes} (scripts/public-budget.json)`]
    : []
  for (const { rel } of sizes) {
    if (!rel.startsWith(LOGOS_DIR)) continue
    problems.push(...logoProblems(rel, fs.readFileSync(path.join(frontendDir, rel))))
  }
  return { total, sizes, problems }
}

function main() {
  const budget = readBudget()
  const files = listPublicFiles(FRONTEND)
  const { total, sizes, problems } = checkPublic(FRONTEND, files, budget)
  const logos = files.filter((rel) => rel.startsWith(LOGOS_DIR)).length
  console.log(`public/: ${files.length} files, ${total} bytes (budget ${budget.max_bytes}); ${logos} logos checked`)
  for (const [top, bytes] of breakdown(sizes)) console.log(`  ${String(bytes).padStart(10)}  ${top}`)
  if (problems.length) {
    console.error(`\n${problems.length} problem(s):`)
    for (const problem of problems) console.error(`  - ${problem}`)
    process.exit(1)
  }
  console.log('OK: within budget, and every logo is a real PNG or SVG.')
}

if (require.main === module) {
  try {
    main()
  } catch (err) {
    console.error(`check-public-assets: ${err.message}`)
    process.exit(1)
  }
}

module.exports = { sniff, pngSize, logoProblems, listPublicFiles, checkPublic, readBudget, breakdown, MAX_LOGO_SIDE }
