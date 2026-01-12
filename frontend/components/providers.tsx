
'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { MockProvider } from '../lib/mock-context'
import { ThemeProvider } from './theme-provider'

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // 1 minute
        retry: 1,
      },
    },
  }))

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      storageKey="automatos-theme"
      disableTransitionOnChange
    >
      <QueryClientProvider client={queryClient}>
        <MockProvider>
          {children}
        </MockProvider>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
