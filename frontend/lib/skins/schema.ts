// PRD-259 §3 — the skin format, automatos-skin/v1. Studio owns this schema; the backend's
// services/skins/schema.py holds the same lists and the shared corpus
// (orchestrator/tests/fixtures/skins/cases.json) proves the two agree. A skin is data:
// token values, shipped font keys, numbers and catalogue names. Never CSS, script or a URL.

export const SKIN_SCHEMA_ID = 'automatos-skin/v1'

/** The tokens a skin may set, taken from app/globals.css (§3). Order is the contract. */
export const ALLOWED_TOKENS = [
  'background', 'foreground', 'card', 'card-foreground', 'popover', 'popover-foreground',
  'primary', 'primary-foreground', 'secondary', 'secondary-foreground', 'muted', 'muted-foreground',
  'accent', 'accent-foreground', 'destructive', 'destructive-foreground', 'border', 'input', 'ring',
  'success', 'success-foreground', 'warning', 'warning-foreground', 'info', 'info-foreground',
  'agent', 'agent-foreground',
  'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5',
  'stage-1', 'stage-2', 'stage-3', 'stage-4', 'stage-5', 'stage-6', 'stage-7', 'stage-8', 'stage-9',
  'olive', 'warn-ink', 'on-strong', 'border-2', 'accent-2', 'navy',
] as const

export type SkinToken = (typeof ALLOWED_TOKENS)[number]

/** Pre-wrapped Studio aliases: Studio derives them from the core tokens; a skin never sets one. */
export const DERIVED_ALIASES = ['paper', 'bg-1', 'bg-2', 'fg', 'fg-2', 'fg-3'] as const

export const AA_TEXT = 4.5
export const AA_NON_TEXT = 3

export interface ContrastPair {
  readonly foreground: SkinToken
  readonly background: SkinToken
  readonly min: number
}

const textPair = (foreground: SkinToken, background: SkinToken): ContrastPair => ({ foreground, background, min: AA_TEXT })

/** Checked on the skin's own values when it sets either side (decision K5). */
export const CONTRAST_PAIRS: readonly ContrastPair[] = [
  textPair('foreground', 'background'),
  textPair('card-foreground', 'card'),
  textPair('popover-foreground', 'popover'),
  textPair('primary-foreground', 'primary'),
  textPair('secondary-foreground', 'secondary'),
  textPair('muted-foreground', 'background'),
  textPair('muted-foreground', 'card'),
  textPair('accent-foreground', 'accent'),
  textPair('destructive-foreground', 'destructive'),
  textPair('success-foreground', 'success'),
  textPair('warning-foreground', 'warning'),
  textPair('info-foreground', 'info'),
  textPair('agent-foreground', 'agent'),
  { foreground: 'ring', background: 'background', min: AA_NON_TEXT },
]

/** The faces app/layout.tsx loads through next/font. */
export const SHIPPED_FONTS = ['geist', 'geist-mono', 'newsreader'] as const
export const FONT_SLOTS = ['sans', 'serif', 'mono'] as const
export const FONT_FILE_PATTERN = /^assets\/fonts\/[A-Za-z0-9_-]{1,64}\.woff2$/

export const MASCOT_MODES = ['off', 'auto', 'custom'] as const
/** The 12 clip states of Play's lib/mascot/types.ts. */
export const MASCOT_STATES = [
  'hello', 'idle', 'listening', 'thinking', 'working', 'searching',
  'reading', 'celebrate', 'oops', 'sleeping', 'lost', 'loading',
] as const
export const MASCOT_SHELLS = ['bronze', 'gold', 'silver', 'cobalt'] as const
export const MASCOT_LIGHTS = ['mint', 'amber', 'ice', 'rose'] as const

export const SKIN_BASES = ['studio', 'classic'] as const
export const SKIN_TONES = ['light', 'dark'] as const

export const RADIUS_MIN = 0
export const RADIUS_MAX = 24
export const GLOW_MIN = 0
export const GLOW_MAX = 40

export const SLUG_MAX = 64
export const SLUG_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/
export const VERSION_MAX = 64
export const VERSION_PATTERN = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$/
export const PREVIEW_PATH = 'assets/preview.png'

/** Free-text limits in code points; `min` 1 means the field can't be empty. */
export const TEXT_LIMITS = {
  name: { min: 1, max: 40 },
  author: { min: 0, max: 80 },
  description: { min: 0, max: 280 },
  licence: { min: 0, max: 80 },
} as const
export type TextField = keyof typeof TEXT_LIMITS

/** Refused anywhere in free text, compared case-insensitively. Control characters are refused too. */
export const TEXT_FORBIDDEN = ['<', '>', '://', 'url(', 'javascript:'] as const

/** A skin's id in Studio (the marketplace_skins row): a UUID. */
export const SKIN_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export const SKIN_FIELDS = [
  'schema', 'slug', 'name', 'version', 'author', 'description', 'licence',
  'base', 'tones', 'tokens', 'fonts', 'shape', 'mascot', 'preview',
] as const
export const REQUIRED_FIELDS = ['schema', 'slug', 'name', 'version', 'base', 'tones', 'tokens'] as const

export type SkinBase = (typeof SKIN_BASES)[number]
export type SkinTone = (typeof SKIN_TONES)[number]
export type FontSlot = (typeof FONT_SLOTS)[number]
export type MascotMode = (typeof MASCOT_MODES)[number]
export type MascotShell = (typeof MASCOT_SHELLS)[number]
export type MascotLight = (typeof MASCOT_LIGHTS)[number]

/** A validated skin: token values are normalised triplets ('H S% L%'). */
export interface Skin {
  readonly schema: typeof SKIN_SCHEMA_ID
  readonly slug: string
  readonly name: string
  readonly version: string
  readonly author?: string
  readonly description?: string
  readonly licence?: string
  readonly base: SkinBase
  readonly tones: readonly SkinTone[]
  readonly tokens: Partial<Record<SkinTone, Readonly<Partial<Record<SkinToken, string>>>>>
  readonly fonts?: Readonly<Partial<Record<FontSlot, string>>>
  readonly shape?: { readonly radius?: number; readonly glow?: number }
  readonly mascot?: { readonly mode: MascotMode; readonly shell?: MascotShell; readonly light?: MascotLight }
  readonly preview?: typeof PREVIEW_PATH
}

export const SKIN_ERROR_CODES = [
  'SCHEMA', 'UNKNOWN_FIELD', 'UNKNOWN_TOKEN', 'DERIVED_TOKEN', 'BAD_VALUE', 'TONES_MISMATCH',
  'PAIR_INCOMPLETE', 'CONTRAST', 'BAD_FONT', 'BAD_SHAPE', 'BAD_MASCOT', 'BAD_TEXT',
] as const
export type SkinErrorCode = (typeof SKIN_ERROR_CODES)[number]

export interface SkinError {
  readonly code: SkinErrorCode
  readonly path: string
  readonly message: string
}

export type SkinValidation =
  | { readonly ok: true; readonly skin: Skin }
  | { readonly ok: false; readonly errors: readonly SkinError[] }
