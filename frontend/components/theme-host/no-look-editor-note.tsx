import { PageHeader } from '@/components/shared/page-header'

/**
 * PRD-261 A-19f1 — the core's note on the pages only a theme draws (Make it yours at /looks, the
 * style sheet at /looks/style), shown where no theme's screen is: under Classic and Studio, under
 * an installed theme that registered no such slot, and below a theme's width (PRD-260 D7). It
 * replaces builtin Play's note: there is no theme to switch to here, so it offers no switch and
 * points at the Appearance menu. An installed theme with the slot (lookEditor@1, styleSheet@1)
 * renders its own screen instead (components/theme-host/theme-slot-page.tsx picks which).
 *
 * The note wears the core's own frame, the shared PageHeader (Studio's editorial head under
 * Studio, the classic head under Classic: PRD-244), and is a region named by the page, so its
 * words read as one block. A theme's screen marks its h1 with THEME_HEADING_ATTRIBUTE
 * (tabIndex={-1}); ThemeSlotPage hands that heading the focus once the screen replaces the note
 * (THEME_HEADING_SELECTOR, P260-FX-22, G3). The note's region carries the same mark, so the page
 * has one focus anchor whichever shows: a switch that reloads onto a theme without this slot
 * hands the focus to the note instead of leaving it on <body>.
 */

/** The mark a theme's screen puts on its h1 (Play's package does since K-19c). */
export const THEME_HEADING_ATTRIBUTE = 'data-theme-heading'
/** The page's heading, as ThemeSlotPage looks for it once a theme's screen (or this note) shows. */
export const THEME_HEADING_SELECTOR = `[${THEME_HEADING_ATTRIBUTE}]`
/** The mark and the focusability a focus anchor carries. */
const HEADING_MARK = { [THEME_HEADING_ATTRIBUTE]: true, tabIndex: -1 } as const

export const LOOKS_TITLE = 'Make it yours'
export const LOOKS_NOTE = 'This style has no look to edit. Choose a theme with looks in the Appearance menu to make it yours.'
export const STYLE_SHEET_TITLE = 'Style sheet'
export const STYLE_SHEET_NOTE = 'This style has no style sheet. Choose a theme in the Appearance menu to see its own.'
export const WIDER_SCREEN_NOTE = "It opens on a laptop or desktop screen: this theme's pages need a wider window."

export interface NoLookEditorNoteProps {
  /** The page's title, also the region's name. */
  readonly title: string
  /** Why the page has no screen here. */
  readonly note: string
}

/** PageHeader's two-part title: the words before the last, then the last word as its accent. */
function titleParts(title: string): { readonly lead: string; readonly accent: string } {
  const words = title.trim().split(/\s+/)
  return { lead: words.slice(0, -1).join(' '), accent: words[words.length - 1] ?? '' }
}

export function NoLookEditorNote({ title, note }: NoLookEditorNoteProps) {
  const { lead, accent } = titleParts(title)
  return (
    <section aria-label={title} {...HEADING_MARK}>
      <PageHeader title={lead} titleAccent={accent} subtitle={note} />
    </section>
  )
}
