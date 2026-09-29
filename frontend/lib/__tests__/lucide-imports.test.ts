/**
 * Every icon imported from lucide-react exists in the installed version (issue #818).
 *
 * `Route` is not exported by lucide-react 0.279.0. The build only warned
 * ("Attempted import error"), the import resolved to undefined, and the Knowledge
 * Graph explorer crashed the moment it rendered <Route />. This guard reads every
 * source file's lucide-react import list and checks each name against the package.
 */
import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import * as lucide from 'lucide-react'

const FRONTEND = path.resolve(__dirname, '../..')
const LUCIDE_TYPES = path.join(FRONTEND, 'node_modules', 'lucide-react', 'dist', 'lucide-react.d.ts')
const SKIP_DIRS = new Set(['node_modules', '.next', 'out', 'coverage'])
const LUCIDE_IMPORT = /import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*['"]lucide-react['"]/g

function sourceFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return SKIP_DIRS.has(entry.name) ? [] : sourceFiles(full)
    return /\.(ts|tsx)$/.test(entry.name) ? [full] : []
  })
}

function importedIcons(source: string): string[] {
  return [...source.matchAll(LUCIDE_IMPORT)].flatMap((match) =>
    match[1]
      .replace(/\/\/.*$/gm, '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .split(',')
      .map((name) => name.trim().replace(/^type\s+/, '').split(/\s+as\s+/)[0])
      .filter(Boolean),
  )
}

// Type-only exports (LucideIcon, LucideProps…) have no runtime value; importing them is fine.
function typeExports(): Set<string> {
  const declarations = fs.readFileSync(LUCIDE_TYPES, 'utf8')
  return new Set([...declarations.matchAll(/\b(?:type|interface)\s+([A-Z]\w*)/g)].map((m) => m[1]))
}

describe('lucide-react imports', () => {
  it('every imported icon is exported by the installed lucide-react', () => {
    const exported = lucide as Record<string, unknown>
    const types = typeExports()
    const missing = sourceFiles(FRONTEND).flatMap((file) =>
      importedIcons(fs.readFileSync(file, 'utf8'))
        .filter((name) => exported[name] === undefined && !types.has(name))
        .map((name) => `${path.relative(FRONTEND, file)}: ${name}`),
    )
    expect(missing).toEqual([])
  })
})
