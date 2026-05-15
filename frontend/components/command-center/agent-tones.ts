/**
 * Agent tones — maps an agent's name (case-insensitive) to a swatch color.
 * Used for kanban swatches, calendar event borders, attention/roster dots.
 *
 * Falls back to a neutral muted tone when an agent is unknown. Add new
 * agents here as the roster grows; the colour values are lifted from CD's
 * round-4 spec.
 */

export interface AgentTone {
  bg: string
  fg: string
}

const TONE_MAP: Record<string, AgentTone> = {
  comms:      { bg: 'hsl(213 51% 35%)', fg: '#fff' },
  qa:         { bg: 'hsl(28 70% 41%)',  fg: '#fff' },
  sentinel:   { bg: 'hsl(15 76% 44%)',  fg: '#fff' },
  scribe:     { bg: 'hsl(38 78% 27%)',  fg: '#fff' },
  scout:      { bg: 'hsl(82 30% 33%)',  fg: '#fff' },
  atlas:      { bg: 'hsl(272 30% 42%)', fg: '#fff' },
  fixer:      { bg: 'hsl(158 44% 33%)', fg: '#fff' },
  vector:     { bg: 'hsl(201 44% 30%)', fg: '#fff' },
  pulse:      { bg: 'hsl(327 49% 36%)', fg: '#fff' },
  auto:       { bg: 'hsl(30 14% 12%)',  fg: '#fff' },
  watchtower: { bg: 'hsl(0 0% 33%)',    fg: '#fff' },
}

const DEFAULT_TONE: AgentTone = {
  bg: 'hsl(var(--muted-foreground))',
  fg: '#fff',
}

function normalize(name: string | null | undefined): string {
  if (!name) return ''
  return name
    .toLowerCase()
    .replace(/\s+engineer$/, '')
    .replace(/\s+/g, '')
    .replace(/[^a-z0-9]/g, '')
}

export function toneFor(agentName: string | null | undefined): AgentTone {
  const key = normalize(agentName)
  if (!key) return DEFAULT_TONE
  // Try direct lookup first
  if (TONE_MAP[key]) return TONE_MAP[key]
  // Try first-word match
  const firstWord = key.split(/\d/)[0]
  if (TONE_MAP[firstWord]) return TONE_MAP[firstWord]
  return DEFAULT_TONE
}

export function initialFor(agentName: string | null | undefined): string {
  if (!agentName) return '?'
  const trimmed = agentName.trim()
  return (trimmed[0] || '?').toUpperCase()
}
