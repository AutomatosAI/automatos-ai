#!/usr/bin/env node
/**
 * Code shape and apiClient-only calls on CHANGED frontend lines (AGENTS.md → Code shape).
 *
 * The full ESLint run in CI is report-only: the codebase carries pre-existing lint
 * problems, and failing on all of them would block every pull request. This gate is
 * the ratchet beside it. It lints only the .ts/.tsx files a change touches, and fails
 * only on findings that sit on the change's own lines, for these rules:
 *
 *   max-lines-per-function  50 code lines, 150 for a React component (a capitalised name)
 *   max-depth               4
 *   no-restricted-syntax    the repository's own rules: apiClient instead of fetch('/api…'),
 *                           and design tokens instead of `orange-*` (.eslintrc.json)
 *
 * It also checks size: a NEW file has at most 800 lines, and an existing file already over
 * that may change, but growing it is a warning. Test files are exempt from length and
 * depth (a describe() callback is long by design).
 *
 * Usage (from `frontend/`):
 *   node scripts/check-changed-lines-eslint.js [base]     # base defaults to origin/main
 *
 * Output is GitHub annotations, so failures show inline on the pull request. Exit 1 on
 * any error.
 */
'use strict'

const path = require('path')
const fs = require('fs')
const { execFileSync } = require('child_process')

const MAX_FUNCTION_LINES = 50
const MAX_COMPONENT_LINES = 150
const MAX_DEPTH = 4
const MAX_FILE_LINES = 800
const SHAPE_RULES = new Set(['max-lines-per-function', 'max-depth'])
const ENFORCED_RULES = new Set([...SHAPE_RULES, 'no-restricted-syntax'])
const TEST_FILE = /(^|\/)__tests__\/|\.(test|spec)\.tsx?$/
const HUNK = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/

// git's stderr is captured, not printed: a new file has no base version, and the
// `fatal: path … not in <base>` that `git show` prints for it is expected, not an error.
function git(args, cwd) {
  return execFileSync('git', args, { cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'] })
}

/** New-side line numbers per file, from `git diff -U0` output. */
function changedLines(diffText) {
  const result = new Map()
  let current = null
  for (const line of diffText.split('\n')) {
    if (line.startsWith('+++ ')) {
      const target = line.slice(4)
      current = target.startsWith('b/') ? target.slice(2) : null
      if (current) result.set(current, new Set())
      continue
    }
    const match = HUNK.exec(line)
    if (match && current) {
      const start = Number(match[1])
      const count = match[2] === undefined ? 1 : Number(match[2])
      for (let n = start; n < start + count; n += 1) result.get(current).add(n)
    }
  }
  return result
}

function overlaps(message, touched) {
  const end = message.endLine || message.line
  for (let n = message.line; n <= end; n += 1) if (touched.has(n)) return true
  return false
}

/** A long function is fine up to the component limit when its name is capitalised. */
function withinComponentAllowance(message) {
  if (message.ruleId !== 'max-lines-per-function') return false
  const name = /'([^']+)'/.exec(message.message)
  const lines = /\((\d+)\)/.exec(message.message)
  return Boolean(name && lines && /^[A-Z]/.test(name[1]) && Number(lines[1]) <= MAX_COMPONENT_LINES)
}

function isFinding(message, relPath, touched) {
  if (!ENFORCED_RULES.has(message.ruleId)) return false
  if (SHAPE_RULES.has(message.ruleId) && TEST_FILE.test(relPath)) return false
  if (!overlaps(message, touched)) return false
  return !withinComponentAllowance(message)
}

function fileSizeFinding(relPath, isNew, linesBefore, linesAfter) {
  if (linesAfter <= MAX_FILE_LINES) return null
  if (isNew) return { level: 'error', file: relPath, line: 1, text: `new file has ${linesAfter} lines (max ${MAX_FILE_LINES}). Split it by concern.` }
  if (linesAfter > linesBefore) {
    return { level: 'warning', file: relPath, line: 1, text: `file grew to ${linesAfter} lines (limit ${MAX_FILE_LINES}); split it instead of growing it.` }
  }
  return null
}

function lineCount(root, ref, relPath) {
  try {
    return git(['show', `${ref}:${relPath}`], root).split('\n').length - 1
  } catch {
    return 0
  }
}

async function lintChanged(root, touchedByFile) {
  const { ESLint } = require('eslint')
  const eslint = new ESLint({
    cwd: path.join(root, 'frontend'),
    overrideConfig: {
      rules: {
        'max-lines-per-function': ['error', { max: MAX_FUNCTION_LINES, skipBlankLines: true, skipComments: true, IIFEs: true }],
        'max-depth': ['error', MAX_DEPTH],
      },
    },
  })
  const files = [...touchedByFile.keys()].map((rel) => path.join(root, rel))
  const results = files.length ? await eslint.lintFiles(files) : []
  return results.flatMap((result) => {
    const rel = path.relative(root, result.filePath)
    return result.messages
      .filter((message) => isFinding(message, rel, touchedByFile.get(rel) || new Set()))
      .map((message) => ({ level: 'error', file: rel, line: message.line, text: `${message.message} (${message.ruleId})` }))
  })
}

async function main() {
  const base = process.argv[2] || 'origin/main'
  const root = git(['rev-parse', '--show-toplevel']).trim()
  const mergeBase = git(['merge-base', 'HEAD', base], root).trim()
  const spec = ['--', 'frontend/*.ts', 'frontend/*.tsx']
  const added = new Set(git(['diff', '--name-only', '--diff-filter=A', mergeBase, 'HEAD', ...spec], root).split('\n').filter(Boolean))
  const touchedByFile = changedLines(git(['diff', '-U0', '--diff-filter=AMR', mergeBase, 'HEAD', ...spec], root))
  const findings = []
  for (const rel of touchedByFile.keys()) {
    const after = fs.readFileSync(path.join(root, rel), 'utf8').split('\n').length - 1
    const size = fileSizeFinding(rel, added.has(rel), lineCount(root, mergeBase, rel), after)
    if (size) findings.push(size)
  }
  findings.push(...(await lintChanged(root, touchedByFile)))
  for (const f of findings) console.log(`::${f.level} file=${f.file},line=${f.line}::${f.text}`)
  const errors = findings.filter((f) => f.level === 'error').length
  console.log(`code shape: ${errors} error(s) on changed frontend lines (functions ≤${MAX_FUNCTION_LINES} lines, components ≤${MAX_COMPONENT_LINES}, depth ≤${MAX_DEPTH}, new files ≤${MAX_FILE_LINES})`)
  return errors ? 1 : 0
}

module.exports = { changedLines, overlaps, withinComponentAllowance, isFinding, fileSizeFinding }

if (require.main === module) {
  main().then((code) => process.exit(code), (error) => {
    console.error(error)
    process.exit(1)
  })
}
