'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ChatbotInterface } from '@/components/chatbot/chatbot-interface'
import { usePageAPI } from '@/hooks/use-page-api'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function ChatPage() {
  usePageAPI('chat')
  
  return (
    <MainLayout>
      <ChatbotInterface />
    </MainLayout>
  )
}
