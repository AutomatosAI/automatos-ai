
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
import { Toaster } from 'react-hot-toast'
import { GlobalSearch } from './shared/global-search'

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
          colorPrimary: '#ff6b35',
          colorBackground: '#0f0f0f',
          colorText: '#fafafa',
          colorInputBackground: '#1a1a1a',
          colorInputText: '#fafafa',
          colorTextSecondary: '#a6a6a6',
          colorDanger: '#e54545',
          fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif',
          borderRadius: '1rem',
        },
        elements: {
          card: 'glass-card card-glow',
          headerTitle: 'text-foreground',
          headerSubtitle: 'text-muted-foreground',
          socialButtonsBlockButton: 'bg-secondary/50 hover:bg-secondary/70 text-foreground border border-border/50 backdrop-blur rounded-2xl',
          formFieldInput: 'bg-background/50 border border-border/50 text-foreground placeholder:text-muted-foreground backdrop-blur rounded-2xl',
          formButtonPrimary: 'bg-primary hover:bg-primary/90 text-primary-foreground rounded-2xl',
          footerActionLink: 'text-primary hover:text-primary/80',
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
            themes={['light', 'dark', 'matte']}
            storageKey="automatos-theme"
            disableTransitionOnChange
          >
            <QueryClientProvider client={queryClient}>
              <WorkspaceProvider>
                <MockProvider>
                  <FirstLoginGuard />
                  {children}
                  <GlobalSearch />
                  <Toaster
                    position="top-right"
                    toastOptions={{
                      style: {
                        background: 'hsl(var(--card))',
                        color: 'hsl(var(--foreground))',
                        border: '1px solid hsl(var(--border))',
                      },
                    }}
                  />
                </MockProvider>
              </WorkspaceProvider>
            </QueryClientProvider>
          </ThemeProvider>
        </RoleProvider>
      </ClerkApiClientProvider>
    </ClerkProvider>
  )
}
