import { useState, useEffect } from 'react'
import { useIsMobile } from './use-mobile'

export type ViewMode = 'grid' | 'list'

export function useViewMode(pageKey: string, defaultMode: ViewMode = 'grid'): [ViewMode, (m: ViewMode) => void] {
  const storageKey = `automatos-view-${pageKey}`
  const isMobile = useIsMobile()

  const [mode, setMode] = useState<ViewMode>(() => {
    if (typeof window !== 'undefined') {
      // Check if user has explicitly set a preference
      const stored = localStorage.getItem(storageKey) as ViewMode | null
      if (stored) return stored
    }
    return defaultMode
  })

  // Auto-switch to list view on mobile (unless user explicitly chose grid)
  useEffect(() => {
    if (isMobile) {
      const hasExplicitChoice = localStorage.getItem(storageKey)
      if (!hasExplicitChoice) {
        setMode('list')
      }
    }
  }, [isMobile, storageKey])

  const setModeAndStore = (m: ViewMode) => {
    setMode(m)
    localStorage.setItem(storageKey, m)
  }

  return [mode, setModeAndStore]
}
