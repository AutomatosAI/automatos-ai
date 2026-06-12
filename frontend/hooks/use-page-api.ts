/**
 * Simple hook to tag API requests with the active page name.
 *
 * Historically this drove per-page mock selection; the mock system was removed
 * in PRD-168 S3, so it now only records which page is making requests.
 */

'use client'

import { useEffect } from 'react'
import { apiClient } from '@/lib/api-client'

/**
 * Configure API behavior for the current page.
 *
 * @param pageName - Name of the page (e.g., 'dashboard', 'agents', 'workflows')
 *
 * @example
 * ```tsx
 * export default function DashboardPage() {
 *   usePageAPI('dashboard')
 * }
 * ```
 */
export function usePageAPI(pageName: string) {
  useEffect(() => {
    apiClient.setCurrentPage(pageName)

    // Cleanup: clear page name on unmount
    return () => {
      apiClient.setCurrentPage('')
    }
  }, [pageName])
}
