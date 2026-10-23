'use client'

// One line, the component first: app/__tests__/page-frames.test.ts follows it to the note's PageHeader.
import { NoLookEditorNote, STYLE_SHEET_NOTE, STYLE_SHEET_TITLE, THEME_HEADING_SELECTOR, WIDER_SCREEN_NOTE } from '@/components/theme-host/no-look-editor-note'
import { ThemeSlotPage } from '@/components/theme-host/theme-slot-page'

/**
 * PRD-260 §4, D8; PRD-261 A-04b (PA-14), A-19f1 — the live style sheet at /looks/style: the active
 * theme's style sheet (styleSheet@1), through the same ThemeSlotPage as Make it yours at /looks,
 * so the two pages behave alike. Under a theme at its width: the sheet on the active look, full
 * bleed inside MainLayout (the theme's frame). Under Classic or Studio, or a theme with no style
 * sheet: the same kind of note as /looks (components/theme-host/no-look-editor-note.tsx), never a
 * redirect, so no loop, and it follows a style change on this page at once. Under a theme below
 * its width (D7): the same 'wider screen' note as /looks.
 */
export default function LookStylePage() {
  return (
    <ThemeSlotPage
      slot="styleSheet@1"
      note={<NoLookEditorNote title={STYLE_SHEET_TITLE} note={STYLE_SHEET_NOTE} />}
      narrowNote={<NoLookEditorNote title={STYLE_SHEET_TITLE} note={WIDER_SCREEN_NOTE} />}
      headingSelector={THEME_HEADING_SELECTOR}
    />
  )
}
