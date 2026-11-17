/**
 * PRD-251D US-D301 (E10) — a post's own look (pure): light or dark, the accent strong or sparing,
 * and the logo on or off. Which of them a template takes comes from its gallery entry (`looks`):
 * light or dark only for a template built on paper; the accent and the logo for every template,
 * "Let Auto pick" included. A choice that is the kit's own (light, the brand kit's accent, the
 * logo on) is left out of the look, so a post that changes nothing saves no look at all. The
 * server checks the look again (modules/socials/post_look.py).
 */
import type { SocialPost, SocialTemplateSummary } from '@/lib/api-client'

export type PostLook = NonNullable<SocialPost['look']>
export type LookKey = NonNullable<SocialTemplateSummary['looks']>[number]
export type ModeValue = 'light' | 'dark'
export type AccentValue = 'kit' | 'bold' | 'sparing'
export type LogoValue = 'on' | 'off'

/** The choices every template takes: the accent and the logo. */
const EVERY_TEMPLATE: ReadonlyArray<LookKey> = ['accent', 'logo']
/** The order a look's keys are kept in (the server's), so two equal looks compare equal as text. */
const LOOK_ORDER: ReadonlyArray<LookKey> = ['mode', 'accent', 'logo']

/** A copy of `look` (a post's, or null) with its keys in the server's order; `{}` for none. */
export function orderedLook(look: PostLook | null | undefined): PostLook {
  const source: PostLook = look ?? {}
  return Object.fromEntries(LOOK_ORDER.filter((key) => key in source).map((key) => [key, source[key]])) as PostLook
}

/** The look choices to show: the chosen template's (`looks`); with none chosen ("Let Auto pick"),
 * or an entry that names none, the accent and the logo. */
export function lookKeys(template: Pick<SocialTemplateSummary, 'looks'> | null): ReadonlyArray<LookKey> {
  return template?.looks?.length ? template.looks : EVERY_TEMPLATE
}

/** The segment each choice shows as pressed: a choice left out is the kit's own. */
export function lookSegments(look: PostLook): { mode: ModeValue; accent: AccentValue; logo: LogoValue } {
  return { mode: look.mode ?? 'light', accent: look.accent ?? 'kit', logo: look.logo === false ? 'off' : 'on' }
}

/** The look with one choice made (a segment's value: 'light' or 'dark'; 'kit', 'bold' or 'sparing';
 * 'on' or 'off'): a choice that is the kit's own is left out, never stored. */
export function withLookChoice(look: PostLook, key: LookKey, value: string): PostLook {
  const rest = Object.fromEntries(Object.entries(look).filter(([name]) => name !== key)) as PostLook
  if (key === 'mode') return orderedLook(value === 'dark' ? { ...rest, mode: 'dark' } : rest)
  if (key === 'accent') return orderedLook(value === 'bold' || value === 'sparing' ? { ...rest, accent: value } : rest)
  return orderedLook(value === 'off' ? { ...rest, logo: false } : rest)
}

/** The look as a save sends it: null when every choice is the kit's own. */
export function lookToSave(look: PostLook): PostLook | null {
  return Object.keys(look).length > 0 ? { ...look } : null
}
