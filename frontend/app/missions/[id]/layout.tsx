import type { ReactNode } from 'react'
import { ThemeSlot } from '@/components/theme-host/theme-slot'

/** The route a theme's mission page is registered on: one segment, any mission (plan §2.6 Q3: missions). */
const MISSION_ROUTE = '/missions/[id]'

/**
 * PRD-261 A-04d — one Mission's page through ThemeSlot. A theme that registers a /missions/[id]
 * page draws it instead at its width, for every mission. The core page reads no query parameter,
 * so no link keeps it. With no theme, or a theme that registers no such page, the core page
 * renders unchanged.
 */
export default function MissionLayout({ children }: { children: ReactNode }) {
  return <ThemeSlot page={MISSION_ROUTE}>{children}</ThemeSlot>
}
