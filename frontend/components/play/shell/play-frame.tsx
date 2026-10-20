'use client'

import { type ReactNode, type RefObject, useRef } from 'react'
import { Search } from 'lucide-react'
import { HeaderUtilities, openGlobalSearch } from '@automatos/theme-sdk'

import { AsleepGate } from '../edges/asleep-gate'
import { PlaySidebar } from './play-sidebar'
import styles from './play-frame.module.css'

/**
 * PRD-260 T-004 — Play's frame around every page under Play at >= 1024 px (D7,
 * D8): Play's sidebar, a header row that keeps the core's utilities (search,
 * Help, notifications, the theme menu, the profile control by edition) and the
 * page. Studio's pages render inside it as they do in the Studio shell, recoloured
 * by the look (D3); Play's own pages bring their own columns. MainLayout renders
 * it, and hands over the floating Auto widget as the Studio shell has it.
 * PRD-261 A-14f: the utilities are the SDK's HeaderUtilities (the Studio header's
 * cluster, PRD §4.4) and search is the SDK's openGlobalSearch.
 */

export interface PlayFrameProps {
  readonly children: ReactNode
  /** As in the Studio shell: no padding and no width cap, a column the page fills (the chat). */
  readonly fullBleed?: boolean
  /** The floating Auto widget, exactly as the Studio shell renders it. */
  readonly autoWidget?: ReactNode
}

/**
 * PRD-260 T-009 — while GET /health fails, AsleepGate shows "Automatos is
 * asleep" in place of the frame and its page; they mount again, fresh, when the
 * backend answers (Ember mock 07, the whole window), and the page's <main>
 * takes the focus the swap dropped (tabIndex={-1}: script focus only;
 * P260-FX-22, M21).
 */
export function PlayFrame(props: PlayFrameProps) {
  const main = useRef<HTMLElement>(null)
  return (
    <AsleepGate wakeFocus={main}>
      <FrameLayout {...props} mainRef={main} />
    </AsleepGate>
  )
}

interface FrameLayoutProps extends PlayFrameProps {
  readonly mainRef: RefObject<HTMLElement>
}

function FrameLayout({ children, fullBleed = false, autoWidget, mainRef }: FrameLayoutProps) {
  // `safe-bottom` (the one safe-area helper, globals.css) keeps the frame's
  // bottom edge clear of a home bar, as the Studio shell does.
  return (
    <div className={`${styles.frame} safe-bottom`}>
      <PlaySidebar />
      <div className={styles.column}>
        <PlayHeader />
        {fullBleed ? (
          <main ref={mainRef} className={styles.fullBleed} tabIndex={-1}>
            {children}
          </main>
        ) : (
          <main ref={mainRef} className={styles.page} tabIndex={-1}>
            <div className={styles.pageInner}>{children}</div>
          </main>
        )}
      </div>
      {autoWidget}
    </div>
  )
}

/** The header row: search (the ⌘K dialog) and the host's header utilities (the Studio header's cluster). */
function PlayHeader() {
  return (
    <header className={styles.header}>
      <button type="button" className={styles.search} onClick={openGlobalSearch} aria-label="Open search (⌘K)">
        <Search className={styles.searchIcon} aria-hidden="true" />
        <span>Search</span>
        <kbd className={styles.kbd}>⌘K</kbd>
      </button>
      <div className={styles.utilities}>
        <HeaderUtilities />
      </div>
    </header>
  )
}
