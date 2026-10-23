
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

/**
 * The page the floating Auto widget names, by the route's first matching prefix (in this order).
 * PRD-260 M70: Play's own pages too (Conversations, at /x/conversations since PRD-261 A-04b; Make
 * it yours and its style sheet).
 */
const PAGE_BY_ROUTE: ReadonlyArray<readonly [string, string]> = [
  ['/agents', 'agents'],
  ['/documents', 'documents'],
  ['/tools', 'tools'],
  ['/marketplace', 'marketplace'],
  ['/analytics', 'analytics'],
  ['/activity', 'activity'],
  ['/command-center', 'activity'],
  ['/assignments', 'assignments'],
  ['/context', 'context'],
  ['/playbooks', 'playbooks'],
  ['/workspace', 'workspace'],
  ['/deliverables', 'workspace'],
  ['/settings', 'settings'],
  ['/team', 'team'],
  ['/chat', 'chat'],
  ['/conversations', 'chat'],
  ['/x/conversations', 'chat'],
  ['/looks', 'looks'],
]
const HOME_PAGE = 'chat'
const OTHER_PAGE = 'dashboard'

function currentPageOf(pathname: string | null): string {
  if (pathname === '/') return HOME_PAGE
  return PAGE_BY_ROUTE.find(([prefix]) => pathname?.startsWith(prefix))?.[1] ?? OTHER_PAGE
}

interface ClassicShellProps extends DesktopShellProps {
  isMobileLayout: boolean
  isStudio: boolean
}

/**
 * The classic chrome: the collapsible sidebar on a desktop, and below 1024 px the sheet menu
 * every style uses (Studio's labelled rail in it, or the classic MobileSidebar).
 */
function ClassicShell({ children, fullBleed, autoWidget, isMobileLayout, isStudio }: ClassicShellProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const handleMenuClick = () => {
    if (isMobileLayout) {
      setMobileMenuOpen(true)
    } else {
      setSidebarCollapsed(!sidebarCollapsed)
    }
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

export function MainLayout({ children, fullBleed = false }: MainLayoutProps) {
  // Known on a browser mount's first render (false on the server and while hydrating): a client
  // navigation on a phone never takes the desktop branch below for a render, where the theme host
  // would hand an installed theme's frame to ThemeFrameSlot (M69).
  const isMobileLayout = useIsTabletOrBelow()
  const isStudio = useIsStudio()
  const themeLayout = useThemeLayout()
  const pathname = usePathname()

  // Auto Widget — floating assistant on every page except /chat, in every shell
  const autoWidget = (
    <AutoWidget position="bottom-right" currentPage={currentPageOf(pathname)} visible={!pathname?.startsWith('/chat')} />
  )

  // PRD-260 D7, PRD-261 A-04b: under a theme, from its entry's minWidth up, the theme's frame
  // (frame@1) draws through ThemeFrameSlot, handed to it by the theme host; below it, the fallback
  // chrome as for Studio. Under a theme while the width isn't known (the server, hydration),
  // nothing: a page mounted in one chrome and again in the other runs its effects twice
  // (P260-FX-25, M69). ThemeFrameSlot draws nothing while the host loads an installed theme, for
  // the same reason.
  if (themeLayout === 'pending') return null

  const studioDesktop = isStudio && !isMobileLayout
  const chrome = studioDesktop ? (
    <StudioDesktopShell fullBleed={fullBleed} autoWidget={autoWidget}>{children}</StudioDesktopShell>
  ) : (
    <ClassicShell fullBleed={fullBleed} autoWidget={autoWidget} isMobileLayout={isMobileLayout} isStudio={isStudio}>
      {children}
    </ClassicShell>
  )
  if (themeLayout !== 'theme' && !studioDesktop) return chrome

  return (
    <ThemeFrameSlot fallback={chrome} fullBleed={fullBleed} autoWidget={autoWidget}>
      {children}
    </ThemeFrameSlot>
  )
}
