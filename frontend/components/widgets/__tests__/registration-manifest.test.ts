/**
 * Every built widget must be registered — the manifest cannot silently lag.
 *
 * Widgets self-register as an import side effect (`registerWidget(...Def)` at
 * module scope), and `components/workspace/index.ts` is the hand-maintained
 * manifest of those imports. PRD-163's MissionApprovalWidget and PRD-193's
 * ToolApprovalWidget were built, tested, and exported — but never added to
 * the manifest, so neither EVER rendered in production. The first
 * confirmation-gated destructive action a client pushed through chat
 * (delete agent, 2026-08-06) dead-ended on "Unknown widget type:
 * tool_approval" instead of an Approve/Deny card.
 *
 * This test closes the class: it scans the widgets directory for every
 * self-registering definition, imports ONLY the manifest (exactly what the
 * app bundle does via the workspace components), and asserts every scanned
 * type is registered. Building a widget without wiring it now fails CI.
 */
import { describe, it, expect } from 'vitest'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

// The manifest — the same side-effect imports the real app performs.
import '@/components/workspace'

import { isWidgetRegistered, getRegisteredTypes } from '../registry'

const WIDGETS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..')

// Scan each widget folder's index.tsx for self-registering definitions.
// (Line comment on purpose: the glob "star-slash" sequence inside a block
// comment terminates it early — esbuild read the rest as code and the whole
// suite failed at transform. See PR #621.)
function scanSelfRegisteringTypes(): { type: string; module: string }[] {
  const found: { type: string; module: string }[] = []
  for (const entry of readdirSync(WIDGETS_DIR)) {
    const indexPath = join(WIDGETS_DIR, entry, 'index.tsx')
    try {
      if (!statSync(join(WIDGETS_DIR, entry)).isDirectory()) continue
      const src = readFileSync(indexPath, 'utf-8')
      if (!src.includes('registerWidget(')) continue
      const m = src.match(/type:\s*['"]([a-z0-9_]+)['"]/)
      if (m) found.push({ type: m[1], module: entry })
    } catch {
      continue // no index.tsx — not a widget module
    }
  }
  return found
}

describe('widget registration manifest', () => {
  const scanned = scanSelfRegisteringTypes()

  it('finds the widget definitions (sanity: scan is not empty)', () => {
    expect(scanned.length).toBeGreaterThanOrEqual(10)
  })

  it.each(scanned)(
    'importing the manifest registers $type ($module)',
    ({ type, module }) => {
      expect(
        isWidgetRegistered(type as never),
        `${module} self-registers '${type}' but components/workspace/index.ts ` +
          `never imports it — the widget is built but can never render. ` +
          `Registered: [${getRegisteredTypes().join(', ')}]`
      ).toBe(true)
    }
  )

  it('the approval cards specifically are registered (the 2026-08-06 regression)', () => {
    expect(isWidgetRegistered('tool_approval' as never)).toBe(true)
    expect(isWidgetRegistered('mission_approval' as never)).toBe(true)
  })
})
