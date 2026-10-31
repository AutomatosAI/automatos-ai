/**
 * PRD-268 Wave K (SDK 1.4.0) — the 1.3.0 surface is unchanged: a 1.x only adds (packages/theme-sdk/CHANGELOG.md), so
 * a theme built against 1.3.0 keeps working on this host.
 *
 * support/sdk-1.3.0-surface.json is SDK 1.3.0's surface as main shipped it (684ea7ef4, #1375): each area's runtime
 * names and the types its .d.ts declared.
 * - Every 1.3.0 runtime name is still planned in its area's surface file, exported by the host's SDK module and
 *   declared as a value; every 1.3.0 type is still declared in its area's .d.ts.
 * - What 1.4.0 adds is exactly its new names: one runtime name (proactive.useRateBrief) and the types of a message's
 *   parts, the brief's speaking row and its thumbs.
 * - A theme's 1.3.0 engines range (`^1.3.0`) admits this host's SDK, and BriefVoice is still placed with no props.
 */
import { readFileSync } from 'fs'
import path from 'path'
import type { ReactElement } from 'react'
import { describe, expect, it } from 'vitest'

import { satisfiesSdkRange } from '@/lib/theme-host/engines'

import * as Runtime from '../index'

interface AreaSurface {
  readonly values: readonly string[]
  readonly types: readonly string[]
}

interface SurfaceSnapshot {
  readonly version: string
  readonly areas: Readonly<Record<string, AreaSurface>>
}

const FRONTEND = path.resolve(__dirname, '../../..')
const SURFACE_DIR = path.join(FRONTEND, 'lib/theme-sdk/surface')
const AREAS_DIR = path.join(FRONTEND, 'packages/theme-sdk/areas')
const SNAPSHOT = JSON.parse(readFileSync(path.join(__dirname, 'support/sdk-1.3.0-surface.json'), 'utf8')) as SurfaceSnapshot
const DECLARED_VALUE = /^export declare (?:const|function|class|let|var) ([A-Za-z_$][\w$]*)/gm
const DECLARED_TYPE = /^export (?:declare )?(?:type|interface) ([A-Za-z_$][\w$]*)/gm

/** What 1.4.0 adds, area by area: nothing else may appear, and nothing of 1.3.0 may go. */
const ADDED_IN_1_4_0: Readonly<Record<string, AreaSurface>> = {
  chat: { values: [], types: ['HostBriefPart', 'HostMessagePart', 'HostReplyChipsPart', 'HostSuggestionPart', 'HostTextPart'] },
  proactive: { values: ['useRateBrief'], types: ['BriefRated', 'BriefRating', 'RateBriefInput'] },
  voice: { values: [], types: ['BriefVoiceProps'] },
}
const NOTHING_ADDED: AreaSurface = { values: [], types: [] }
/** The host's SDK module by name. */
const RUNTIME_NAMES = Runtime as unknown as Readonly<Record<string, unknown>>

/** A theme on 1.3.0 places BriefVoice with no props: the host's must still take that call (tsc holds this line). */
const PLACED_AS_IN_1_3_0: () => ReactElement = Runtime.BriefVoice

function declarationOf(area: string): string {
  return readFileSync(path.join(AREAS_DIR, `${area.replace('/', '-')}.d.ts`), 'utf8')
}

function planned(area: string): readonly string[] {
  const file = path.join(SURFACE_DIR, `${area.replace('/', '-')}.json`)
  return (JSON.parse(readFileSync(file, 'utf8')) as { names: readonly string[] }).names
}

function captured(text: string, pattern: RegExp): string[] {
  return Array.from(new Set(Array.from(text.matchAll(pattern), (match) => match[1]))).sort()
}

const missing = (kept: readonly string[], now: readonly string[]) => kept.filter((name) => !now.includes(name))
const added = (before: readonly string[], now: readonly string[]) => now.filter((name) => !before.includes(name)).sort()
const AREAS = Object.keys(SNAPSHOT.areas).map((area) => [area] as const)

describe('SDK 1.4.0 — the 1.3.0 surface unchanged', () => {
  it('reads the 1.3.0 snapshot', () => {
    expect(SNAPSHOT.version).toBe('1.3.0')
    expect(AREAS.length).toBe(18)
  })

  it.each(AREAS)('%s still plans, exports and declares every 1.3.0 runtime name', (area) => {
    const { values } = SNAPSHOT.areas[area]
    expect(missing(values, planned(area))).toEqual([])
    expect(values.filter((name) => RUNTIME_NAMES[name] === undefined)).toEqual([])
    expect(missing(values, captured(declarationOf(area), DECLARED_VALUE))).toEqual([])
  })

  it.each(AREAS)('%s still declares every 1.3.0 type', (area) => {
    expect(missing(SNAPSHOT.areas[area].types, captured(declarationOf(area), DECLARED_TYPE))).toEqual([])
  })

  it.each(AREAS)('%s adds exactly what 1.4.0 adds', (area) => {
    const before = SNAPSHOT.areas[area]
    const expected = ADDED_IN_1_4_0[area] ?? NOTHING_ADDED
    expect(added(before.values, planned(area))).toEqual(expected.values)
    expect(added(before.types, captured(declarationOf(area), DECLARED_TYPE))).toEqual(expected.types)
  })
})

describe('SDK 1.4.0 — a 1.3.0 theme on this host', () => {
  it('is admitted: the host is 1.4.0, and ^1.3.0 accepts it', () => {
    expect(Runtime.SDK_VERSION).toBe('1.4.0')
    expect(satisfiesSdkRange('^1.3.0', Runtime.SDK_VERSION)).toBe(true)
  })

  it('places BriefVoice with no props, as 1.3.0 declared it', () => {
    expect(PLACED_AS_IN_1_3_0).toBe(Runtime.BriefVoice)
  })
})
