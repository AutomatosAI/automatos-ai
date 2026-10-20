import { Suspense, type ReactNode } from 'react'
import { AiLabelGate } from '@/components/ai-label/ai-label'
import { DiscussionBar } from '@/components/chatbot/discussion-bar'
import { ThemeSlot } from '@/components/theme-host/theme-slot'
import { CHAT_HREF } from '@/lib/ticket-links'

/** Links into a surface only the core chat has: the Code canvas (repo), a session's runtime, plan mode. */
const CORE_CHAT_PARAMS = ['repo', 'runtime', 'mode']

/**
 * The chat page, with the bar a ticket's Discuss shows over it (PRD-252 R2). A theme that
 * registers a /chat page (builtin Play: its chat, PRD-260 K7) draws it instead of the page at its
 * width, through ThemeSlot (PRD-261 A-04b); a link with one of CORE_CHAT_PARAMS keeps the core page.
 * PRD-264 EUAI-11a: the gate draws the host's AI label after the page, outside ThemeSlot (the
 * theme's page slot), so the default chat (with Mission Mode), Play's chat and any theme's /chat
 * all show it, and none draws a second copy.
 */
export default function ChatLayout({ children }: { children: ReactNode }) {
  return (
    <AiLabelGate>
      <ThemeSlot page={CHAT_HREF} keepCorePage={CORE_CHAT_PARAMS}>
        {children}
      </ThemeSlot>
      <Suspense fallback={null}>
        <DiscussionBar />
      </Suspense>
    </AiLabelGate>
  )
}
