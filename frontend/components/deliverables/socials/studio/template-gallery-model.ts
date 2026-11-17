/**
 * PRD-251D US-D107 (E7) — what the Look card's template gallery shows, as plain functions.
 *
 * Each template says what it is for (its description). A search over the name and the
 * description, and the chips All, With a photo and No photo, narrow the list in the
 * browser: a format has a few dozen templates at most, so nothing is asked per keystroke.
 * The list keeps the server's order (by name). "Let Auto pick" is not a template, so the
 * gallery shows it first whatever the filter. The chosen template stays shown, in its own
 * place, when the search or a chip would hide it, so the person never loses what they picked.
 */
import type { SocialTemplateSummary } from '@/lib/api-client'

export type TemplateFilter = 'all' | 'photo' | 'no_photo'

/** The chips under the search box. */
export const TEMPLATE_FILTERS: ReadonlyArray<{ value: TemplateFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'photo', label: 'With a photo' },
  { value: 'no_photo', label: 'No photo' },
]
/** The chips' group name, for a screen reader. */
export const FILTER_LABEL = 'Show templates'
export const SEARCH_LABEL = 'Search templates'
export const SEARCH_PLACEHOLDER = 'Search by name or by what it is for'

/** How the no-match line names the chip that is on (All names none). */
const FILTER_WORDS: Record<TemplateFilter, string> = {
  all: '',
  photo: ' with a photo',
  no_photo: ' without a photo',
}

/** The part of a template the gallery searches and filters on. */
export type GalleryTemplate = Pick<SocialTemplateSummary, 'id' | 'name' | 'description' | 'image_slots'>

/** A template shows a picture when it has an image slot. The Photo badge and the chips both use this rule. */
export function hasPhoto(template: Pick<SocialTemplateSummary, 'image_slots'>): boolean {
  return (template.image_slots ?? []).length > 0
}

/** The words of a search, lower-cased; a search of only spaces has none. */
function searchWords(search: string): string[] {
  return search.toLowerCase().split(/\s+/).filter(Boolean)
}

/** Every word of the search is in the template's name or its description, in any case. */
export function matchesSearch(template: Pick<GalleryTemplate, 'name' | 'description'>, search: string): boolean {
  const text = `${template.name} ${template.description ?? ''}`.toLowerCase()
  return searchWords(search).every((word) => text.includes(word))
}

/** With a photo keeps the templates that show a picture, No photo the rest, All every one. */
export function matchesFilter(template: Pick<GalleryTemplate, 'image_slots'>, filter: TemplateFilter): boolean {
  if (filter === 'all') return true
  return filter === 'photo' ? hasPhoto(template) : !hasPhoto(template)
}

export interface GalleryQuery {
  search: string
  filter: TemplateFilter
  /** The chosen template's id; null when Auto picks. */
  chosen: string | null
}

export interface GalleryView<T> {
  /** The templates to show, in the server's order: those that match, and the chosen one. */
  shown: ReadonlyArray<T>
  /** The search or a chip narrowed a gallery that has templates, and none of them matches. */
  noMatch: boolean
}

/** What the gallery shows for a search and a chip. The chosen template is shown, not counted as a match. */
export function galleryView<T extends GalleryTemplate>(templates: ReadonlyArray<T>, query: GalleryQuery): GalleryView<T> {
  const matches = (template: T) => matchesFilter(template, query.filter) && matchesSearch(template, query.search)
  const narrowed = query.filter !== 'all' || searchWords(query.search).length > 0
  return {
    shown: templates.filter((template) => matches(template) || template.id === query.chosen),
    noMatch: narrowed && templates.length > 0 && !templates.some(matches),
  }
}

/** What the gallery says when no template matches: the search's words, and the chip when it is not All. */
export function noMatchLine(search: string, filter: TemplateFilter): string {
  const words = search.trim()
  const which = FILTER_WORDS[filter]
  return words ? `No template${which} matches "${words}".` : `There is no template${which} for this format.`
}
