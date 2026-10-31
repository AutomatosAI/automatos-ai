/**
 * PRD-271 C02 (D-1, host 1.2.0) — a `mobile` block's `layout` and `slots`, checked: one of Automatos Go's phone
 * layouts, and the slots its contract lists. The TypeScript twin of `orchestrator/services/themes/mobile_layout.py`
 * (`layout_errors`), `layout_slots.py` (the slot kinds) and `layout_contracts.py` (the table). The contracts are
 * `phone-layouts.json`, the host's `services/themes/data/phone-layouts.json` less each layout's `home` (no runtime
 * file of the frontend names a first-party theme's id, PRD-261 A-20; the browser never needs it), which
 * `__tests__/validator-parity.test.ts` holds equal; the shared corpus's `manifests` cases hold both languages to
 * the same verdicts.
 *
 * - `layout` and `slots` come together; `layout` names a contract the host knows, `"<id>@<major>"`;
 * - `slots` is an object: no slot the contract lacks, every slot it requires (one with no `default`), each value a
 *   literal of its slot's kind or `{"look": …}`: a variable the browser's first paint keeps (colour and tint slots)
 *   or a look knob (number and choice slots);
 * - kinds: `colour` `#rrggbb` or `H S% L%`; `tint` a colour, `rgba(r,g,b,a)` (no spaces, channels 0 to 255, the
 *   alpha 0 to 1 with up to four decimals) or `transparent`; `number` a whole number in range; `choice` one of its
 *   choices.
 *
 * Each refusal is BAD_MOBILE at `mobile.layout` or `mobile.slots.<name>`. The engines rule (host 1.2.0) is
 * mobile-manifest.ts's. Nothing here throws on input of the wrong shape.
 */
import {
  echo, hasOwn, isKnobName, isRecord, isWholeNumber, PREPAINT_PATH, themeError, type JsonRecord, type ThemeError,
} from './css-value'
import { isMobileColour } from './mobile-colour'
import phoneLayoutsTable from './phone-layouts.json'

export const LAYOUT_FIELD = 'layout'
export const SLOTS_FIELD = 'slots'
export const LAYOUT_FIELDS: readonly string[] = [LAYOUT_FIELD, SLOTS_FIELD]
export const LAYOUT_PATTERN = /^([a-z][a-z0-9-]{0,31})@([1-9][0-9]{0,3})$/
export const SLOT_KINDS: readonly string[] = ['colour', 'tint', 'number', 'choice']
export const LOOK_KEY = 'look'
/**
 * A variable the browser's first paint sets: boot-sanitize.ts CSS_VAR_NAME_PATTERN (mobile_tokens.py BOOT_VAR_NAME),
 * written here because boot-sanitize.ts reaches package-manifest.ts, which imports this file; the parity test holds
 * the two equal.
 */
export const LOOK_VARIABLE_PATTERN = /^--[a-z0-9-]{1,64}$/
const MOBILE_ERROR_CODE = 'BAD_MOBILE'
const TRANSPARENT = 'transparent'
export const RGBA_PATTERN = /^rgba\(([0-9]{1,3}),([0-9]{1,3}),([0-9]{1,3}),(?:0|1|0?\.[0-9]{1,4}|1\.0{1,4})\)$/
const CHANNEL_MAX = 255
const LOOK_VARIABLE = '{"look": "--variable"}'
const LOOK_KNOB = '{"look": "knob"}'

export type SlotKind = 'colour' | 'tint' | 'number' | 'choice'
/** One slot of a contract: its kind, its schematic's group, its range or choices, and a default when optional. */
export interface LayoutSlot {
  readonly kind: SlotKind
  readonly group: string
  readonly min?: number
  readonly max?: number
  readonly choices?: readonly string[]
  readonly default?: string | number
}
/** A slot as a theme fills it: a literal of its kind, or a variable or knob of the member's look. */
export type LayoutSlotValue = string | number | { readonly look: string }
export type LayoutPair = readonly [foreground: string, background: string, kind: 'text' | 'ui']

/** One major of one layout (layout_contracts.py LayoutContract, less its home theme). */
export interface LayoutContract {
  readonly layout: string
  readonly layoutId: string
  readonly major: number
  readonly name: string
  readonly revision: number
  readonly tone: { readonly rule: 'ground' | 'light' | 'dark'; readonly ground?: string; readonly text?: string }
  readonly slots: Readonly<Record<string, LayoutSlot>>
  readonly fixed: Readonly<Record<string, string>>
  readonly pairs: readonly LayoutPair[]
  readonly fonts: Readonly<Record<string, string>>
  readonly mascot: 'bundled' | 'drawn'
  readonly example: Readonly<Record<string, string | number>>
}

type ContractVersion = Omit<LayoutContract, 'layout' | 'layoutId' | 'major' | 'name'>
interface PhoneLayout {
  readonly name: string
  readonly versions: Readonly<Record<string, ContractVersion>>
}
interface PhoneLayouts {
  readonly schema: string
  readonly layouts: Readonly<Record<string, PhoneLayout>>
}

/** The host's contracts table, as the browser's copy holds it (no `home`). */
export const PHONE_LAYOUTS = phoneLayoutsTable as unknown as PhoneLayouts

/** The contract `"<id>@<major>"` names, or null for anything else (an unknown id or major, a wrong shape). */
export function contractOf(layout: unknown): LayoutContract | null {
  const match = typeof layout === 'string' ? LAYOUT_PATTERN.exec(layout) : null
  if (match === null) return null
  const [, layoutId, major] = match
  const entry = hasOwn(PHONE_LAYOUTS.layouts, layoutId) ? PHONE_LAYOUTS.layouts[layoutId] : undefined
  const version = entry !== undefined && hasOwn(entry.versions, major) ? entry.versions[major] : undefined
  if (entry === undefined || version === undefined) return null
  return { ...version, layout: `${layoutId}@${major}`, layoutId, major: Number(major), name: entry.name }
}

/** Every contract the host knows, as a theme names it (`"play@1"`), in the table's order. */
export function layoutNames(): string[] {
  return Object.entries(PHONE_LAYOUTS.layouts).flatMap(([layoutId, entry]) => (
    Object.keys(entry.versions).map((major) => `${layoutId}@${major}`)
  ))
}

/** A colour, an `rgba(r,g,b,a)` whose channels are 0 to 255, or `transparent` (layout_slots.py is_tint). */
export function isTint(value: unknown): value is string {
  if (typeof value !== 'string') return false
  if (value === TRANSPARENT || isMobileColour(value)) return true
  const match = RGBA_PATTERN.exec(value)
  return match !== null && match.slice(1, 4).every((channel) => Number(channel) <= CHANNEL_MAX)
}

/** True when a literal fits the slot's kind (layout_slots.py slot_value is not None). */
export function fitsSlot(spec: LayoutSlot, value: unknown): boolean {
  if (spec.kind === 'colour') return isMobileColour(value)
  if (spec.kind === 'tint') return isTint(value)
  if (spec.kind === 'number') return isWholeNumber(value) && value >= (spec.min ?? 0) && value <= (spec.max ?? 0)
  return typeof value === 'string' && (spec.choices ?? []).includes(value)
}

/** The name in `{"look": "<name>"}` (that one key, its value text), else null. */
export function lookRef(value: unknown): string | null {
  if (!isRecord(value)) return null
  const keys = Object.keys(value)
  const name = keys.length === 1 && keys[0] === LOOK_KEY ? value[LOOK_KEY] : undefined
  return typeof name === 'string' ? name : null
}

/** A colour or tint slot comes from a prepaint variable; a number or choice slot from a look knob. */
export function takesVariable(spec: LayoutSlot): boolean {
  return spec.kind === 'colour' || spec.kind === 'tint'
}

/** `name` is a variable (colour and tint slots) or a knob (number and choice slots) the first paint keeps. */
export function isLookSource(spec: LayoutSlot, name: unknown): boolean {
  if (takesVariable(spec)) return typeof name === 'string' && LOOK_VARIABLE_PATTERN.test(name)
  return isKnobName(name)
}

function bad(path: string, message: string): ThemeError[] {
  return [themeError(MOBILE_ERROR_CODE, path, message)]
}

/** What a slot of this kind takes. */
function kindSentence(where: string, spec: LayoutSlot): string {
  if (spec.kind === 'colour') return `"${where}" must be a colour ("#rrggbb" or "H S% L%") or ${LOOK_VARIABLE}.`
  if (spec.kind === 'tint') return `"${where}" must be a colour, "rgba(r,g,b,a)" or "transparent", or ${LOOK_VARIABLE}.`
  if (spec.kind === 'number') return `"${where}" must be a whole number from ${spec.min} to ${spec.max}, or ${LOOK_KNOB}.`
  const choices = (spec.choices ?? []).map((choice) => `"${choice}"`).join(', ')
  return `"${where}" must be one of ${choices}, or ${LOOK_KNOB}.`
}

/** What `{"look": …}` names for a slot of this kind. */
function sourceSentence(where: string, spec: LayoutSlot): string {
  if (takesVariable(spec)) {
    return (
      `"${where}" takes ${LOOK_VARIABLE}: a variable ${PREPAINT_PATH} writes, "--" then 1 to 64 lower-case `
      + 'letters, digits or dashes.'
    )
  }
  return `"${where}" takes ${LOOK_KNOB}: a look knob's name, a letter then up to 31 letters, digits or _.`
}

function valueErrors(name: string, spec: LayoutSlot, value: unknown): ThemeError[] {
  const where = `mobile.slots.${name}`
  if (isRecord(value)) return isLookSource(spec, lookRef(value)) ? [] : bad(where, sourceSentence(where, spec))
  return fitsSlot(spec, value) ? [] : bad(where, kindSentence(where, spec))
}

/** No slot the contract lacks, every one it requires, each value of its kind or from the look. */
function slotsErrors(contract: LayoutContract, slots: unknown): ThemeError[] {
  if (!isRecord(slots)) {
    return bad('mobile.slots', `"mobile.slots" must be an object of ${contract.layout}'s slots and their values.`)
  }
  const unknown = Object.keys(slots).filter((name) => !hasOwn(contract.slots, name)).map((name) => themeError(
    MOBILE_ERROR_CODE, `mobile.slots.${echo(name)}`, `"${echo(name)}" is not a slot of ${contract.layout}.`,
  ))
  const known = Object.entries(contract.slots).flatMap(([name, spec]) => {
    if (hasOwn(slots, name)) return valueErrors(name, spec, slots[name])
    if (spec.default !== undefined) return []
    return bad(`mobile.slots.${name}`, `"mobile.slots" needs "${name}": ${contract.layout} has no default for it.`)
  })
  return [...unknown, ...known]
}

/** Every reason the block's `layout` and `slots` are refused; empty without them (mobile_layout.py layout_errors). */
export function layoutErrors(block: JsonRecord): ThemeError[] {
  const hasLayout = hasOwn(block, LAYOUT_FIELD)
  const hasSlots = hasOwn(block, SLOTS_FIELD)
  if (!hasLayout && !hasSlots) return []
  if (!hasSlots) return bad(`mobile.${SLOTS_FIELD}`, `"mobile.layout" needs "mobile.slots": the values of the layout's slots.`)
  if (!hasLayout) {
    return bad(`mobile.${LAYOUT_FIELD}`, (
      '"mobile.slots" needs "mobile.layout": the layout and contract major its slots fill, such as "candy@1".'
    ))
  }
  const contract = contractOf(block[LAYOUT_FIELD])
  if (contract === null) {
    const known = layoutNames().map((name) => `"${name}"`).join(', ')
    return bad(`mobile.${LAYOUT_FIELD}`, `"mobile.layout" must name a phone layout and its contract major: one of ${known}.`)
  }
  return slotsErrors(contract, block[SLOTS_FIELD])
}
