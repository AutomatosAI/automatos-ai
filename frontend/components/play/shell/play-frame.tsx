'use client'

import type { ReactNode } from 'react'
import { Search } from 'lucide-react'

import { StudioHeaderUtilities } from '@/components/layout/studio-header'
import { openGlobalSearch } from '@/lib/global-search-event'

import { PlaySidebar } from './play-sidebar'
import styles from './play-frame.module.css'

/**
 * PRD-260 T-004 — Play's frame around every page under Play at >= 1024 px (D7,
 * D8): Play's sidebar, a header row that keeps the core's utilities (search,
 * Help, notifications, the theme menu, the profile control by edition) and the
 * page. Studio's pages render inside it as they do in the Studio shell, recoloured
 * by the look (D3); Play's own pages bring their own columns. MainLayout renders
 * it, and hands over the floating Auto widget as the Studio shell has it.
 */

export interface PlayFrameProps {
  readonly children: ReactNode
  /** As in the Studio shell: no padding and no width cap, a column the page fills (the chat). */
  readonly fullBleed?: boolean
  /** The floating Auto widget, exactly as the Studio shell renders it. */
  readonly autoWidget?: ReactNode
}

export function PlayFrame({ children, fullBleed = false, autoWidget }: PlayFrameProps) {
  // `safe-bottom` (the one safe-area helper, globals.css) keeps the frame's
  // bottom edge clear of a home bar, as the Studio shell does.
  return (
    <div className={`${styles.frame} safe-bottom`}>
      <PlaySidebar />
      <div className={styles.column}>
        <PlayHeader />
        {fullBleed ? (
          <main className={styles.fullBleed}>{children}</main>
        ) : (
          <main className={styles.page}>
            <div className={styles.pageInner}>{children}</div>
          </main>
        )}
      </div>
      {autoWidget}
    </div>
  )
}

/** The header row: search (the ⌘K dialog) and the Studio header's utilities cluster. */
function PlayHeader() {
  return (
    <header className={styles.header}>
      <button type="button" className={styles.search} onClick={openGlobalSearch} aria-label="Open search (⌘K)">
        <Search className={styles.searchIcon} aria-hidden="true" />
        <span>Search</span>
        <kbd className={styles.kbd}>⌘K</kbd>
      </button>
      <div className={styles.utilities}>
        <StudioHeaderUtilities />
      </div>
    </header>
  )
}
