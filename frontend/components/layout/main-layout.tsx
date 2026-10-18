
'use client'

import { useCallback, useEffect, useState } from 'react'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { Sidebar } from './sidebar'
import { MobileSidebar } from './mobile-sidebar'
import { Header } from './header'
import { StudioSidebar } from './studio-sidebar'
import { StudioHeader } from './studio-header'
import { AutoWidget } from '../chatbot/chat-widget'
import { PlayFrame } from '@/components/play/shell/play-frame-lazy'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { ThemeFrameSlot } from '@/components/theme-host/theme-frame-slot'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useIsStudio, useThemeLayout } from '@/hooks/use-studio-theme'

interface MainLayoutProps {
  children: React.ReactNode
  /**
   * When true, render the main content area with no padding and no max-width
   * cap so the page can take the full available width and height. Use for
   * full-canvas surfaces like chat. Studio shell only — classic layout still
   * applies its own padding.
   */
  fullBleed?: boolean
}

interface DesktopShellProps {
  children: React.ReactNode
  fullBleed: boolean
  /** The floating Auto widget, built once by MainLayout for every shell. */
  autoWidget: React.ReactNode
}

const STUDIO_SIDEBAR_COLLAPSED_KEY = 'studioSidebarCollapsed'

/** The Studio rail's collapsed state, remembered per browser. */
function useStudioSidebarCollapsed(): [boolean, () => void] {
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    try {
      if (localStorage.getItem(STUDIO_SIDEBAR_COLLAPSED_KEY) === '1') setCollapsed(true)
    } catch {
      // Storage blocked (a private window): the rail opens expanded.
    }
  }, [])

  const toggle = useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem(STUDIO_SIDEBAR_COLLAPSED_KEY, next ? '1' : '0')
      } catch {
        // Storage blocked: the rail still toggles, it just isn't remembered.
      }
      return next
    })
  }, [])

  return [collapsed, toggle]
}

// ────────────────────────────────────────────────────────────────────
// Studio shell — render the CD round-4 chrome (sidebar + header)
// when .studio is active and we're on desktop. Mobile keeps the existing
// sheet pattern for now; MainLayout falls through to the classic layout.
// ────────────────────────────────────────────────────────────────────
function StudioDesktopShell({ children, fullBleed, autoWidget }: DesktopShellProps) {
  const [collapsed, toggleCollapsed] = useStudioSidebarCollapsed()
  // `safe-bottom` (the one safe-area helper, globals.css) keeps the shell's
  // bottom edge — the scrolling main and the rail alike — clear of a
  // notched phone's home bar. The shell is 100dvh with border-box sizing,
  // so the inset comes out of its height.
  return (
    <div className="sh-shell safe-bottom">
      <StudioSidebar collapsed={collapsed} onToggle={toggleCollapsed} />
      <div className="sh-main">
        <StudioHeader />
        {fullBleed ? (
          <main className="sh-fullbleed flex-1 min-h-0 flex flex-col">
            {children}
          </main>
        ) : (
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
        )}
      </div>
      {autoWidget}
    </div>
  )
}

export function MainLayout({ children, fullBleed = false }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const isMobileLayout = useIsTabletOrBelow()
  const isStudio = useIsStudio()
  const themeLayout = useThemeLayout()
  const pathname = usePathname()

  // Get current page context for the chat
  const getCurrentPage = () => {
    if (pathname === '/') return 'chat'
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
    // PRD-260 M70: Play's own pages (Conversations; Make it yours and its style sheet).
    if (pathname?.startsWith('/conversations')) return 'chat'
    if (pathname?.startsWith('/looks')) return 'looks'
    return 'dashboard'
  }

  const currentPage = getCurrentPage()
  const showAutoWidget = !pathname?.startsWith('/chat')
  // Auto Widget — floating assistant on every page except /chat, in every shell
  const autoWidget = (
    <AutoWidget position="bottom-right" currentPage={currentPage} visible={showAutoWidget} />
  )

  const handleMenuClick = () => {
    if (isMobileLayout) {
      setMobileMenuOpen(true)
    } else {
      setSidebarCollapsed(!sidebarCollapsed)
    }
  }

  // PRD-260 D7, PRD-261 A-02b: the theme's frame from its entry's minWidth up; below it, the
  // fallback chrome as for Studio. Under a theme while the width isn't known (the server,
  // hydration), nothing: a page mounted in one chrome and again in the other runs its effects
  // twice (P260-FX-25, M69). Builtin Play is the only theme with a layout here until A-04b
  // moves its frame into the frame@1 slot.
  if (themeLayout === 'pending') return null
  if (themeLayout === 'theme') {
    return <PlayFrame fullBleed={fullBleed} autoWidget={autoWidget}>{children}</PlayFrame>
  }

  if (isStudio && !isMobileLayout) {
    const shell = <StudioDesktopShell fullBleed={fullBleed} autoWidget={autoWidget}>{children}</StudioDesktopShell>
    return (
      <ThemeFrameSlot fallback={shell} fullBleed={fullBleed} autoWidget={autoWidget}>
        {children}
      </ThemeFrameSlot>
    )
  }

  // PRD-246: below 1024 the Studio pages render inside this chrome, and they are
  // built for the Studio shell's contract — a definite-height column they fill
  // and scroll inside (`.cc-page` is `height: 100%; overflow-y: auto`, the chat
  // shell `height: 100%; overflow: hidden`). A padded, auto-height <main> gave
  // the chat a 65px shell with nothing scrollable. `fullBleed` now means the
  // same thing here that it means in the shell.
  const bleed = fullBleed && isMobileLayout

  return (
    <div className={bleed ? 'h-[100dvh] flex flex-col overflow-hidden gradient-bg' : 'min-h-screen gradient-bg overflow-x-hidden'}>
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

      {/* Mobile Navigation Sheet — Studio uses the labelled rail, classic uses the existing MobileSidebar */}
      {isMobileLayout && (
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetContent
            side="left"
            className={
              isStudio
                ? 'w-[260px] p-0 bg-secondary border-r border-border safe-bottom'
                : 'w-[280px] p-0 glass-card border-r border-primary/15 bg-background/95 backdrop-blur-lg'
            }
          >
            {isStudio ? (
              <StudioSidebar />
            ) : (
              <MobileSidebar onNavigate={() => setMobileMenuOpen(false)} />
            )}
          </SheetContent>
        </Sheet>
      )}

      {/* Main Content */}
      <div className={
        bleed
          ? 'flex-1 min-h-0 flex flex-col transition-all duration-300'
          : isMobileLayout
            ? 'transition-all duration-300'
            : 'transition-all duration-300 ml-16'
      }>
        <Header onMenuClick={handleMenuClick} />

        <main
          data-bleed={bleed ? 'phone' : undefined}
          className={bleed ? 'flex-1 min-h-0 flex flex-col' : 'px-4 py-4 md:px-6 md:py-6 lg:px-14 lg:py-8 2xl:px-16'}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: isMobileLayout ? 0.2 : 0.5 }}
            className={bleed ? 'flex-1 min-h-0 flex flex-col' : 'max-w-[1720px] mx-auto'}
          >
            {children}
          </motion.div>
        </main>
      </div>

      {autoWidget}
    </div>
  )
}
