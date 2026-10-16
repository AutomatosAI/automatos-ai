// @vitest-environment node
/**
 * PRD-259 S-010b — no `dark:` class paints a fixed palette colour.
 *
 * Since S-010a (`darkMode: ['class', '.dark']`) every `dark:` class renders, in
 * Classic Dark, Studio Dark and a skin's dark tone alike. A fixed neutral
 * (`dark:text-gray-200`, `dark:bg-white`, `dark:border-[#333]`) ignores the tone's
 * tokens and paints over Studio and every skin. Use the token class instead
 * (`text-foreground`, `bg-card`, `border-border`, …), or drop the `dark:` class when
 * the base class is already the token. docs/PRDS/PRD-259-DARK-VARIANT-AUDIT.md has
 * the verdict rules and one row per use.
 */
import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

const FRONTEND = path.resolve(__dirname, '../..')
const SCANNED_DIRS = ['app', 'components', 'lib', 'hooks', 'contexts']
const SKIP_DIRS = new Set(['__tests__', 'node_modules', '.next', 'out', 'coverage'])
const SOURCE_FILE = /\.(ts|tsx)$/

/** `dark:` (any variant chain after it) + a colour utility + a fixed neutral or an arbitrary hex. */
const FIXED_DARK_CLASS = new RegExp(
  String.raw`dark:(?:[a-z-]+:)*` +
    String.raw`(?:bg|text|border|ring|from|to|via|divide|outline|fill|stroke|placeholder)-` +
    String.raw`(?:(?:gray|slate|zinc|neutral|stone)-\d+|white\b|black\b|\[#[0-9a-fA-F]+\])` +
    String.raw`[^\s"'\`]*`,
  'g',
)

const SHEET_ARTIFACT_RATCHET = 'fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines)'
const MESSAGE_RATCHET = 'fix — blocked by the code-shape ratchet (Message, 409 lines)'

/**
 * file:class pairs the audit could not fix: each sits inside a function over the
 * AGENTS.md limit, where the changed-lines gate fails on any touched line. Remove an
 * entry when its function is split and the class moves to a token.
 */
const ALLOWLIST: Record<string, string> = {
  'components/chatbot/sheet-artifact.tsx dark:border-gray-800/60': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/sheet-artifact.tsx dark:text-gray-200': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/sheet-artifact.tsx dark:text-gray-100': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/sheet-artifact.tsx dark:divide-gray-800/70': SHEET_ARTIFACT_RATCHET,
  'components/chatbot/message.tsx dark:text-gray-200': MESSAGE_RATCHET,
}

function sourceFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return SKIP_DIRS.has(entry.name) ? [] : sourceFiles(full)
    return SOURCE_FILE.test(entry.name) ? [full] : []
  })
}

function fixedDarkClasses(source: string): string[] {
  return [...source.matchAll(FIXED_DARK_CLASS)].map((match) => match[0])
}

/** Every `<file> <class>` pair in the scanned tree. */
function foundPairs(): string[] {
  const files = SCANNED_DIRS.flatMap((dir) => sourceFiles(path.join(FRONTEND, dir)))
  return files.flatMap((file) => {
    const rel = path.relative(FRONTEND, file).split(path.sep).join('/')
    return fixedDarkClasses(fs.readFileSync(file, 'utf8')).map((cls) => `${rel} ${cls}`)
  })
}

describe('PRD-259 S-010b — dark: classes use tokens, not fixed colours', () => {
  it('the pattern catches fixed neutrals and hexes, and passes token classes', () => {
    const caught = fixedDarkClasses(
      'dark:text-gray-200 dark:hover:bg-zinc-800/50 dark:border-[#333333] dark:text-white dark:bg-black/40',
    )
    expect(caught).toEqual([
      'dark:text-gray-200',
      'dark:hover:bg-zinc-800/50',
      'dark:border-[#333333]',
      'dark:text-white',
      'dark:bg-black/40',
    ])
    expect(fixedDarkClasses('dark:text-foreground dark:bg-background/40 dark:border-border dark:text-info/80')).toEqual([])
  })

  it('no dark: class outside the allowlist paints a fixed palette colour', () => {
    const offenders = [...new Set(foundPairs())].filter((pair) => !(pair in ALLOWLIST))
    expect(offenders).toEqual([])
  })

  it('every allowlist entry still exists (drop it once its class is fixed)', () => {
    const found = new Set(foundPairs())
    const stale = Object.keys(ALLOWLIST).filter((pair) => !found.has(pair))
    expect(stale).toEqual([])
  })
})
