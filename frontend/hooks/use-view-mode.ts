import { useState, useEffect } from 'react'

export type ViewMode = 'grid' | 'list'

export function useViewMode(pageKey: string, defaultMode: ViewMode = 'grid'): [ViewMode, (m: ViewMode) => void] {
  const storageKey = `automatos-view-${pageKey}`
  const [mode, setMode] = useState<ViewMode>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem(storageKey) as ViewMode) || defaultMode
    }
    return defaultMode
  })

  useEffect(() => {
    localStorage.setItem(storageKey, mode)
  }, [storageKey, mode])

  return [mode, setMode]
}
