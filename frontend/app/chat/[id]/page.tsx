import { notFound } from 'next/navigation'
import { Chat } from '@/components/chatbot/chat'
import { AppSidebar } from '@/components/chatbot/sidebar'
import { getChat, getChatMessages } from '@/lib/chat/api'
import { MainLayout } from '@/components/layout/main-layout'

export default async function ChatDetailPage({ params }: { params: { id: string } }) {
  try {
    const [chat, messages] = await Promise.all([
      getChat(params.id),
      getChatMessages(params.id),
    ])

    return (
      <MainLayout>
        <div className="flex h-[calc(100vh-8rem)]">
          <AppSidebar />
          <div className="flex-1 bg-gray-950 rounded-lg border border-gray-800">
            <Chat
              id={chat.id}
              initialMessages={messages}
              initialChatModel="gpt-4"
              initialVisibilityType={chat.visibility}
              isReadonly={false}
              autoResume={false}
              initialLastContext={chat.lastContext}
            />
          </div>
        </div>
      </MainLayout>
    )
  } catch (error) {
    console.error('Failed to load chat:', error)
    notFound()
  }
}
