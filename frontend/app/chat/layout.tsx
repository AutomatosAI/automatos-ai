import { Suspense, type ReactNode } from 'react'
import { DiscussionBar } from '@/components/chatbot/discussion-bar'

/** The chat page, with the bar a ticket's Discuss shows over it (PRD-252 R2). */
export default function ChatLayout({ children }: { children: ReactNode }) {
  return (
    <>
      {children}
      <Suspense fallback={null}>
        <DiscussionBar />
      </Suspense>
    </>
  )
}
