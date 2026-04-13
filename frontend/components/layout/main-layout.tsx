
'use client'

import { useState } from 'react'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { Sidebar } from './sidebar'
import { MobileSidebar } from './mobile-sidebar'
import { Header } from './header'
import { ChatWidget } from '../chatbot/chat-widget'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { usePageTour } from '@/components/onboarding/use-page-tour'

interface MainLayoutProps {
  children: React.ReactNode
}

export function MainLayout({ children }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const isMobileLayout = useIsTabletOrBelow()
  const pathname = usePathname()

  // Auto-launch page-specific tours on first visit
  usePageTour()

  // Get current page context for the chat
  const getCurrentPage = () => {
    if (pathname === '/') return 'chat'
    if (pathname.startsWith('/dashboard')) return 'dashboard'
    if (pathname.startsWith('/agents')) return 'agents'
    if (pathname.startsWith('/documents')) return 'documents'
    if (pathname.startsWith('/tools')) return 'tools'
    if (pathname.startsWith('/marketplace')) return 'marketplace'
    if (pathname.startsWith('/analytics')) return 'analytics'
    if (pathname.startsWith('/context')) return 'context'
    if (pathname.startsWith('/playbooks')) return 'playbooks'
    if (pathname.startsWith('/workspace')) return 'workspace'
    if (pathname.startsWith('/chat')) return 'chat'
    return 'dashboard'
  }

  const chatContext = {
    currentPage: getCurrentPage(),
    selectedItems: [], // This could be populated based on page state
    userRole: 'user', // This could come from authentication
    recentActions: [] // This could track recent user actions
  }

  const handleMenuClick = () => {
    if (isMobileLayout) {
      setMobileMenuOpen(true)
    } else {
      setSidebarCollapsed(!sidebarCollapsed)
    }
  }

  return (
    <div className="min-h-screen gradient-bg overflow-x-hidden">
      {/* Desktop Sidebar — hidden below lg */}
      {!isMobileLayout && (
        <>
          <Sidebar collapsed={sidebarCollapsed} onToggle={setSidebarCollapsed} />

          {/* Overlay scrim when sidebar expanded on desktop */}
          {!sidebarCollapsed && (
            <div
              className="fixed inset-0 z-30 bg-black/30 backdrop-blur-[1px]"
              onClick={() => setSidebarCollapsed(true)}
              aria-hidden="true"
            />
          )}
        </>
      )}

      {/* Mobile Navigation Sheet */}
      {isMobileLayout && (
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetContent
            side="left"
            className="w-[280px] p-0 glass-card border-r border-primary/15 bg-background/95 backdrop-blur-lg"
          >
            <MobileSidebar onNavigate={() => setMobileMenuOpen(false)} />
          </SheetContent>
        </Sheet>
      )}

      {/* Main Content */}
      <div className={
        isMobileLayout
          ? 'flex flex-col h-screen transition-all duration-300'
          : 'flex flex-col h-screen transition-all duration-300 ml-16'
      }>
        <Header onMenuClick={handleMenuClick} />

        <main className="flex-1 min-h-0 overflow-y-auto p-4 md:p-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: isMobileLayout ? 0.2 : 0.5 }}
            className="max-w-7xl mx-auto h-full"
          >
            {children}
          </motion.div>
        </main>
      </div>

      {/* Pilot Helper Widget — shown on ALL pages (overlay, no navigation) */}
      <ChatWidget
        position="bottom-right"
        context={chatContext}
      />
    </div>
  )
}
