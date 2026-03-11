#!/usr/bin/env npx tsx
/**
 * sync-icon-registry.ts
 *
 * Scans frontend/public/assets/icons/ for SVG files and syncs them with
 * frontend/config/iconRegistry.json.
 *
 * - NEW files on disk → auto-generates registry entry (id, filename, path, name, tags)
 * - EXISTING entries → preserved as-is (your custom tags/names are never overwritten)
 * - ORPHAN entries (registry entry but no SVG on disk) → removed with warning
 *
 * Usage:
 *   npx tsx frontend/scripts/sync-icon-registry.ts          # sync
 *   npx tsx frontend/scripts/sync-icon-registry.ts --dry-run # preview changes
 *   npx tsx frontend/scripts/sync-icon-registry.ts --check   # CI: exit 1 if out of sync
 */

import { readdirSync, readFileSync, writeFileSync, existsSync } from 'fs'
import { join, basename, resolve } from 'path'

// Resolve paths relative to script location
const ROOT = resolve(__dirname, '..')
const ICONS_DIR = join(ROOT, 'public', 'assets', 'icons')
const REGISTRY_PATH = join(ROOT, 'config', 'iconRegistry.json')

interface IconEntry {
  id: string
  filename: string
  path: string
  tags: string[]
  name: string
}

// ── Helpers ──────────────────────────────────────────────────────────

/** Convert filename to human-readable name: "email-all-stacked.svg" → "Email All Stacked" */
function filenameToName(filename: string): string {
  return filename
    .replace(/\.svg$/, '')
    .split(/[-_]/)
    .filter(Boolean)
    .map((word) => {
      // Keep fully-numeric segments as-is (e.g. "3d")
      if (/^\d+\w*$/.test(word)) return word
      return word.charAt(0).toUpperCase() + word.slice(1)
    })
    .join(' ')
}

/** Convert filename to kebab-case id: "Email-All_Stacked.svg" → "email-all-stacked" */
function filenameToId(filename: string): string {
  return filename
    .replace(/\.svg$/, '')
    .toLowerCase()
    .replace(/[_\s]+/g, '-')
}

/** Generate search tags from filename segments + common defaults */
function filenameToTags(filename: string): string[] {
  const segments = filename
    .replace(/\.svg$/, '')
    .toLowerCase()
    .split(/[-_]/)
    .filter((s) => s.length > 1) // skip single chars

  // Deduplicate and add default tags
  const tags = [...new Set([...segments, 'streamline', 'gradient'])]
  return tags
}

/** Create a new registry entry from a filename */
function createEntry(filename: string): IconEntry {
  return {
    id: filenameToId(filename),
    filename,
    path: `/assets/icons/${filename}`,
    tags: filenameToTags(filename),
    name: filenameToName(filename),
  }
}

// ── Main ─────────────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2)
  const dryRun = args.includes('--dry-run')
  const check = args.includes('--check')

  // 1. Read all SVGs from disk
  if (!existsSync(ICONS_DIR)) {
    console.error(`Icon directory not found: ${ICONS_DIR}`)
    process.exit(1)
  }

  const svgFiles = readdirSync(ICONS_DIR)
    .filter((f) => f.endsWith('.svg'))
    .sort()

  const svgSet = new Set(svgFiles)

  // 2. Read existing registry
  let existing: IconEntry[] = []
  if (existsSync(REGISTRY_PATH)) {
    try {
      existing = JSON.parse(readFileSync(REGISTRY_PATH, 'utf-8'))
    } catch (err) {
      console.error(`Failed to parse ${REGISTRY_PATH}:`, err)
      process.exit(1)
    }
  }

  const existingByFilename = new Map<string, IconEntry>()
  for (const entry of existing) {
    existingByFilename.set(entry.filename, entry)
  }

  // 3. Build new registry
  const added: string[] = []
  const removed: string[] = []
  const kept: string[] = []
  const merged: IconEntry[] = []

  // Process each SVG on disk
  for (const svgFile of svgFiles) {
    const existingEntry = existingByFilename.get(svgFile)
    if (existingEntry) {
      // Keep existing entry (preserve custom tags/names)
      merged.push(existingEntry)
      kept.push(svgFile)
    } else {
      // New file — auto-generate entry
      merged.push(createEntry(svgFile))
      added.push(svgFile)
    }
  }

  // Detect orphans (in registry but not on disk)
  for (const entry of existing) {
    if (!svgSet.has(entry.filename)) {
      removed.push(entry.filename)
    }
  }

  // Detect duplicates
  const dupeCount = existing.length - existingByFilename.size
  const hasDuplicates = dupeCount > 0

  // 4. Report
  console.log(`\n📦 Icon Registry Sync`)
  console.log(`   SVGs on disk:     ${svgFiles.length}`)
  console.log(`   Registry entries:  ${existing.length}${hasDuplicates ? ` (${dupeCount} duplicates)` : ''}`)
  console.log(`   ─────────────────────────`)
  console.log(`   ✅ Kept:           ${kept.length}`)

  if (added.length > 0) {
    console.log(`   ➕ Added:          ${added.length}`)
    for (const f of added) {
      console.log(`      + ${f}`)
    }
  }

  if (removed.length > 0) {
    console.log(`   🗑️  Removed:        ${removed.length}`)
    for (const f of removed) {
      console.log(`      - ${f}`)
    }
  }

  console.log(`   ─────────────────────────`)
  console.log(`   📋 Final count:    ${merged.length}\n`)

  if (hasDuplicates) {
    console.log(`   🔄 Deduped:        ${dupeCount}`)
  }

  // 5. Check mode — exit 1 if anything changed
  if (check) {
    if (added.length > 0 || removed.length > 0 || hasDuplicates) {
      console.error('❌ Registry is out of sync. Run: npx tsx frontend/scripts/sync-icon-registry.ts')
      process.exit(1)
    }
    console.log('✅ Registry is in sync.')
    return
  }

  // 6. Dry run — don't write
  if (dryRun) {
    console.log('🔍 Dry run — no changes written.')
    return
  }

  // 7. Write if anything changed
  if (added.length === 0 && removed.length === 0 && !hasDuplicates) {
    console.log('✨ Already in sync — nothing to do.')
    return
  }

  // Sort by id for consistent output
  merged.sort((a, b) => a.id.localeCompare(b.id))

  writeFileSync(REGISTRY_PATH, JSON.stringify(merged, null, 2) + '\n', 'utf-8')
  console.log(`✅ Written to ${REGISTRY_PATH}`)
}

main()
