import { LOOK_INTERFACE, LOOK_SCHEMA, type Look } from './types'

/** PRD-260 — the five presets of Play's Looks prototype (P). Ember is the default. */

type PresetFields = Omit<Look, 'schema' | 'interface'>

function preset(fields: PresetFields): Look {
  return { schema: LOOK_SCHEMA, interface: LOOK_INTERFACE, ...fields }
}

export const EMBER: Look = preset({ id: 'ember', name: 'Ember', sub: 'Warm, editorial', ground: 'graphite', accent: 'molten', font: 'serif', shell: 'bronze', light: 'mint', glow: 24 })
export const OBSIDIAN: Look = preset({ id: 'obsidian', name: 'Obsidian', sub: 'Mono, minimal', ground: 'obsidian', accent: 'cream', font: 'clean', shell: 'bronze', light: 'mint', glow: 6 })
export const AURORA: Look = preset({ id: 'aurora', name: 'Aurora', sub: 'Cool and glassy', ground: 'obsidian', accent: 'mint', font: 'bric', shell: 'bronze', light: 'ice', glow: 20 })
export const WORKSHOP: Look = preset({ id: 'workshop', name: 'Workshop', sub: 'The original', ground: 'indigo', accent: 'molten', font: 'round', shell: 'bronze', light: 'mint', glow: 14 })
export const DAYLIGHT: Look = preset({ id: 'daylight', name: 'Daylight', sub: 'For daytime', ground: 'paper', accent: 'molten', font: 'serif', shell: 'bronze', light: 'mint', glow: 10 })

export const PRESETS: readonly Look[] = [EMBER, OBSIDIAN, AURORA, WORKSHOP, DAYLIGHT]

export const DEFAULT_LOOK: Look = EMBER

export function isPresetId(id: string): boolean {
  return PRESETS.some((look) => look.id === id)
}

export function findPreset(id: string): Look | undefined {
  return PRESETS.find((look) => look.id === id)
}
