
'use client'

import { useEffect, useState } from 'react'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { Sidebar } from './sidebar'
import { MobileSidebar } from './mobile-sidebar'
import { Header } from './header'
import { StudioSidebar } from './studio-sidebar'
import { StudioHeader } from './studio-header'
import { StudioTicker } from './studio-ticker'
import { AutoWidget } from '../chatbot/chat-widget'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useAutoTour } from '@/hooks/use-auto-tour'
import { useIsStudio } from '@/hooks/use-studio-theme'

interface MainLayoutProps {
  children: React.ReactNode
}

export function MainLayout({ children }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [studioSidebarCollapsed, setStudioSidebarCollapsed] = useState(false)
  const isMobileLayout = useIsTabletOrBelow()
  const isStudio = useIsStudio()
  const pathname = usePathname()

  useEffect(() => {
    try {
      const stored = localStorage.getItem('studioSidebarCollapsed')
      if (stored === '1') setStudioSidebarCollapsed(true)
    } catch {}
  }, [])

  const toggleStudioSidebar = () => {
    setStudioSidebarCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem('studioSidebarCollapsed', next ? '1' : '0')
      } catch {}
      return next
    })
  }

  // Auto-start page tour on first visit (per-user, per-page)
  useAutoTour()

  // Get current page context for the chat
  const getCurrentPage = () => {
    if (pathname === '/') return 'chat'
    if (pathname.startsWith('/dashboard')) return 'dashboard'
    if (pathname.startsWith('/agents')) return 'agents'
    if (pathname.startsWith('/documents')) return 'documents'
    if (pathname.startsWith('/tools')) return 'tools'
    if (pathname.startsWith('/marketplace')) return 'marketplace'
    if (pathname.startsWith('/analytics')) return 'analytics'
    if (pathname.startsWith('/activity')) return 'activity'
    if (pathname.startsWith('/command-center')) return 'activity'
    if (pathname.startsWith('/assignments')) return 'assignments'
    if (pathname.startsWith('/context')) return 'context'
    if (pathname.startsWith('/playbooks')) return 'playbooks'
    if (pathname.startsWith('/workspace')) return 'workspace'
    if (pathname.startsWith('/deliverables')) return 'workspace'
    if (pathname.startsWith('/settings')) return 'settings'
    if (pathname.startsWith('/team')) return 'team'
    if (pathname.startsWith('/chat')) return 'chat'
    return 'dashboard'
  }

  const currentPage = getCurrentPage()
  const showAutoWidget = !pathname?.startsWith('/chat')

  const handleMenuClick = () => {
    if (isMobileLayout) {
      setMobileMenuOpen(true)
    } else {
      setSidebarCollapsed(!sidebarCollapsed)
    }
  }

  // ────────────────────────────────────────────────────────────────────
  // Studio shell — render the CD round-4 chrome (sidebar + header + ticker)
  // when .studio is active and we're on desktop. Mobile keeps the existing
  // sheet pattern for now. Falls through to classic layout below.
  // ────────────────────────────────────────────────────────────────────
  if (isStudio && !isMobileLayout) {
    return (
      <div className="sh-shell">
        <StudioSidebar
          collapsed={studioSidebarCollapsed}
          onToggle={toggleStudioSidebar}
        />
        <div className="sh-main">
          <StudioTicker />
          <StudioHeader />
          <main className="px-4 py-4 md:px-6 md:py-6 lg:px-12 lg:py-8 2xl:px-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="max-w-[1720px] mx-auto"
            >
              {children}
            </motion.div>
          </main>
        </div>
        <AutoWidget
          position="bottom-right"
          currentPage={currentPage}
          visible={showAutoWidget}
        />
      </div>
    )
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
          ? 'transition-all duration-300'
          : 'transition-all duration-300 ml-16'
      }>
        <Header onMenuClick={handleMenuClick} />

        <main className="px-4 py-4 md:px-6 md:py-6 lg:px-14 lg:py-8 2xl:px-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: isMobileLayout ? 0.2 : 0.5 }}
            className="max-w-[1720px] mx-auto"
          >
            {children}
          </motion.div>
        </main>
      </div>

      {/* Auto Widget — floating assistant on every page except /chat */}
      <AutoWidget
        position="bottom-right"
        currentPage={currentPage}
        visible={showAutoWidget}
      />
    </div>
  )
}
