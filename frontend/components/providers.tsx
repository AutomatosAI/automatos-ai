
'use client'

import { ClerkProvider } from '@clerk/nextjs'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { MockProvider } from '../lib/mock-context'
import { ThemeProvider } from './theme-provider'
import { WorkspaceProvider } from './workspace-provider'
import { ClerkApiClientProvider } from './clerk-api-client-provider'
import { RoleProvider } from '../contexts/role-context'
import { FirstLoginGuard } from './onboarding/first-login-guard'

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
    <ClerkProvider
      appearance={{
        baseTheme: undefined,
        variables: {
          colorPrimary: '#7c3aed',
          colorBackground: '#0f1117',
          colorText: '#e5e7eb',
          colorInputBackground: '#1f2430',
          colorInputText: '#e5e7eb',
          colorTextSecondary: '#9aa4b2',
          colorDanger: '#ef4444',
          fontFamily: 'Inter, ui-sans-serif, system-ui',
        },
        elements: {
          card: 'bg-slate-900/95 border border-slate-800/80 shadow-2xl',
          headerTitle: 'text-white',
          headerSubtitle: 'text-slate-400',
          socialButtonsBlockButton: 'bg-slate-800 hover:bg-slate-700 text-white',
          formFieldInput: 'bg-slate-800 border border-slate-700 text-white placeholder:text-slate-500',
          formButtonPrimary: 'bg-violet-600 hover:bg-violet-500 text-white',
          footerActionLink: 'text-violet-300 hover:text-violet-200',
        },
        layout: {
          logoImageUrl: '/brand/automatos-mark-hi.png',
        },
      }}
    >
      <ClerkApiClientProvider>
        <RoleProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            storageKey="automatos-theme"
            disableTransitionOnChange
          >
            <QueryClientProvider client={queryClient}>
              <WorkspaceProvider>
                <MockProvider>
                  <FirstLoginGuard />
                  {children}
                </MockProvider>
              </WorkspaceProvider>
            </QueryClientProvider>
          </ThemeProvider>
        </RoleProvider>
      </ClerkApiClientProvider>
    </ClerkProvider>
  )
}
