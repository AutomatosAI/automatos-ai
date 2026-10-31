'use client'

// One line, the component first: app/__tests__/page-frames.test.ts follows it to the note's PageHeader.
import { NoLookEditorNote, LOOKS_NOTE, LOOKS_NO_THEME_NOTE, LOOKS_TITLE, THEME_HEADING_SELECTOR, WIDER_SCREEN_NOTE } from '@/components/theme-host/no-look-editor-note'
import { ThemeSlotPage } from '@/components/theme-host/theme-slot-page'

/**
 * PRD-260 §4, K9; PRD-261 A-04b (PA-14), A-19f1 — Make it yours: the active theme's look editor
 * (lookEditor@1), full bleed inside MainLayout (the theme's frame). Under Classic or Studio, or a
 * theme with no look editor: the core's note (components/theme-host/no-look-editor-note.tsx),
 * which offers no switch, and says how to get a theme while the workspace has none to choose. The style comes from the UiStyleProvider, which the server starts from
 * the style cookie, so the first paint is already the right one.
 */
export default function LooksPage() {
  return (
    <ThemeSlotPage
      slot="lookEditor@1"
      note={<NoLookEditorNote title={LOOKS_TITLE} note={LOOKS_NOTE} noThemeNote={LOOKS_NO_THEME_NOTE} />}
      narrowNote={<NoLookEditorNote title={LOOKS_TITLE} note={WIDER_SCREEN_NOTE} />}
      headingSelector={THEME_HEADING_SELECTOR}
    />
  )
}
