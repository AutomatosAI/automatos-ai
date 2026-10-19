import type { ReactNode } from 'react'
import { ThemeSlot } from '@/components/theme-host/theme-slot'

/** The route a theme's knowledge page is registered on (registerPage, plan §2.6 Q3: knowledge, D0). */
const DOCUMENTS_ROUTE = '/documents'

/**
 * PRD-261 A-04d — the Knowledge page (/documents) through ThemeSlot. A theme that registers a
 * /documents page draws it instead at its width. The core page reads no query parameter, so no
 * link keeps it. With no theme, or a theme that registers no /documents page, the core page
 * renders unchanged.
 */
export default function DocumentsLayout({ children }: { children: ReactNode }) {
  return <ThemeSlot page={DOCUMENTS_ROUTE}>{children}</ThemeSlot>
}
