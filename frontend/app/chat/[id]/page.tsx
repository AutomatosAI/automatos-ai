import { notFound } from 'next/navigation'
import { Chat } from '@/components/chatbot/chat'
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
        <div className="h-[calc(100vh-8rem)] flex flex-col bg-gray-950 rounded-lg border border-gray-800">
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
      </MainLayout>
    )
  } catch (error) {
    console.error('Failed to load chat:', error)
    notFound()
  }
}
