
'use client'

import { useState } from 'react'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { Sidebar } from './sidebar'
import { Header } from './header'
import { ChatWidget } from '../chatbot/chat-widget'

interface MainLayoutProps {
  children: React.ReactNode
}

export function MainLayout({ children }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const pathname = usePathname()

  // Get current page context for the chat
  const getCurrentPage = () => {
    if (pathname === '/') return 'dashboard'
    if (pathname.startsWith('/agents')) return 'agents'
    if (pathname.startsWith('/documents')) return 'documents'
    if (pathname.startsWith('/workflows')) return 'workflows'
    if (pathname.startsWith('/analytics')) return 'analytics'
    if (pathname.startsWith('/context')) return 'context'
    if (pathname.startsWith('/playbooks')) return 'playbooks'
    if (pathname.startsWith('/chat')) return 'chat'
    return 'dashboard'
  }

  const chatContext = {
    currentPage: getCurrentPage(),
    selectedItems: [], // This could be populated based on page state
    userRole: 'user', // This could come from authentication
    recentActions: [] // This could track recent user actions
  }

  return (
    <div className="min-h-screen gradient-bg">
      {/* Sidebar */}
      <Sidebar collapsed={sidebarCollapsed} onToggle={setSidebarCollapsed} />
      
      {/* Main Content */}
      <div className={`transition-all duration-300 ${sidebarCollapsed ? 'ml-16' : 'ml-64'}`}>
        <Header onMenuClick={() => setSidebarCollapsed(!sidebarCollapsed)} />
        
        <main className="p-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-7xl mx-auto"
          >
            {children}
          </motion.div>
        </main>
      </div>
      
      {/* Chat Widget - Only show on non-chat pages */}
      {pathname !== '/chat' && (
        <ChatWidget 
          position="bottom-right"
          context={chatContext}
        />
      )}
    </div>
  )
}
