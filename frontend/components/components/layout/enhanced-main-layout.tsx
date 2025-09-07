

'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { usePathname } from 'next/navigation'
import { Sidebar } from './sidebar'
import { Header } from './header'
import { ChatWidget } from '../chatbot/chat-widget'

interface MainLayoutProps {
  children: React.ReactNode
}

export function EnhancedMainLayout({ children }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const pathname = usePathname()

  // Determine current page context for chatbot
  const getCurrentPageContext = () => {
    const path = pathname?.split('/')[1] || 'dashboard'
    
    return {
      currentPage: path,
      selectedItems: [], // TODO: Could be populated from page context
      userRole: 'admin', // TODO: Get from auth context
      recentActions: [] // TODO: Could be populated from activity tracking
    }
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

      {/* NEW: AI Chatbot Widget */}
      <ChatWidget 
        position="bottom-right"
        context={getCurrentPageContext()}
      />
    </div>
  )
}
