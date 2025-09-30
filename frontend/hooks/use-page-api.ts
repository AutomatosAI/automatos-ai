/**
 * Simple hook to configure page-level API behavior
 * 
 * This tells the API client which page is making requests,
 * so it can use the correct mock/real API configuration.
 */

'use client'

import { useEffect } from 'react'
import { apiClient } from '@/lib/api-client'

/**
 * Configure API behavior for the current page
 * 
 * @param pageName - Name of the page (e.g., 'dashboard', 'agents', 'workflows')
 * 
 * @example
 * ```tsx
 * export default function DashboardPage() {
 *   usePageAPI('dashboard')  // That's it!
 *   // Now all API calls respect dashboard config
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

/**
 * Get whether the current page should use mocks
 * 
 * @param pageName - Optional page name (uses current page if not provided)
 * @returns boolean - true if using mocks, false if using real APIs
 */
export function usePageMockStatus(pageName?: string): boolean {
  return apiClient.getPageMockStatus(pageName)
}

/**
 * Temporarily override mock setting for a page (useful for testing)
 * 
 * @example
 * ```tsx
 * // Enable mocks for dashboard temporarily
 * overridePageMocks('dashboard', true)
 * 
 * // Disable mocks
 * overridePageMocks('dashboard', false)
 * ```
 */
export function overridePageMocks(pageName: string, useMocks: boolean) {
  apiClient.setPageMockOverride(pageName, useMocks)
}


