import { Suspense, type ReactNode } from 'react'
import { DiscussionBar } from '@/components/chatbot/discussion-bar'
import { PlayRouteSwitch } from '@/components/play/play-route-switch'
import { PlayChat } from '@/components/play/chat/play-chat-lazy'

/** The chat page under Play at >= 1024 px: Play's chat instead of the page (PRD-260 K7). */
const PLAY_CHAT_PATHS = ['/chat']
/** Links into a surface only the core chat has: the Code canvas (repo), a session's runtime, plan mode. */
const CORE_CHAT_PARAMS = ['repo', 'runtime', 'mode']

/** The chat page, with the bar a ticket's Discuss shows over it (PRD-252 R2). */
export default function ChatLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <PlayRouteSwitch
        paths={PLAY_CHAT_PATHS}
        keepPageFor={CORE_CHAT_PARAMS}
        play={
          <Suspense fallback={null}>
            <PlayChat />
          </Suspense>
        }
      >
        {children}
      </PlayRouteSwitch>
      <Suspense fallback={null}>
        <DiscussionBar />
      </Suspense>
    </>
  )
}
