import { Suspense, type ReactNode } from 'react'
import { AiLabelGate } from '@/components/ai-label/ai-label'
import { AGENT_LINK_PARAM, AgentLinkProvider } from '@/components/chatbot/agent-link'
import { DiscussionBar } from '@/components/chatbot/discussion-bar'
import { ThemeSlot } from '@/components/theme-host/theme-slot'
import { VoiceProvider } from '@/components/voice/voice-provider'
import { CHAT_HREF } from '@/lib/ticket-links'

/**
 * Links into a surface only the core chat has: the Code canvas (repo), a session's runtime, plan mode, and an agent
 * picked by link (a session agent's Claude Code session opens in the core chat's canvas).
 */
const CORE_CHAT_PARAMS = ['repo', 'runtime', 'mode', AGENT_LINK_PARAM]

/**
 * The chat page, with the bar a ticket's Discuss shows over it (PRD-252 R2). A theme that
 * registers a /chat page (installed Play, for one: its chat) draws it instead of the page at its
 * width, through ThemeSlot (PRD-261 A-04b); a link with one of CORE_CHAT_PARAMS keeps the core page.
 * PRD-264 EUAI-11a: the gate draws the host's AI label after the page, outside ThemeSlot (the
 * theme's page slot), so the default chat (with Mission Mode), Play's chat and any theme's /chat
 * all show it, and none draws a second copy.
 * Theme voice: the host voice provider sits inside the gate, around the page's slot, so the default chat and a
 * theme's /chat read the same voice session (Studio's Live strip, a theme's voice controls).
 * SDK 1.5.0: `/chat?agent=<id>` picks that agent in the core chat once (AgentLinkProvider holds the link for the
 * visit; components/chatbot/agent-link.tsx), so a theme's HostLink reaches a session agent's Claude Code session.
 */
export default function ChatLayout({ children }: { children: ReactNode }) {
  return (
    <AiLabelGate>
      <VoiceProvider>
        <AgentLinkProvider>
          <ThemeSlot page={CHAT_HREF} keepCorePage={CORE_CHAT_PARAMS}>
            {children}
          </ThemeSlot>
        </AgentLinkProvider>
      </VoiceProvider>
      <Suspense fallback={null}>
        <DiscussionBar />
      </Suspense>
    </AiLabelGate>
  )
}
