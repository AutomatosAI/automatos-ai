
'use client'

import { ClerkProvider } from '@clerk/nextjs'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Suspense, useState } from 'react'
import { ThemeProvider } from './theme-provider'
import { WorkspaceProvider } from './workspace-provider'
import { ClerkApiClientProvider } from './clerk-api-client-provider'
import { LocalAuthProvider } from './local-auth-provider'
import { RoleProvider } from '../contexts/role-context'
import { Toaster } from './ui/sonner'
import { GlobalSearch } from './shared/global-search'
import { useStudioThemeFlag } from '../hooks/use-studio-theme'
import { isSaaS } from '@/lib/auth-edition'

// Inline child so we can read useSearchParams (needs a Suspense boundary in Next).
function StudioThemeFlag() {
  useStudioThemeFlag()
  return null
}

// PRD-175 (F008): the auth boundary is edition-conditional and is the ONLY thing
// that differs between editions — everything below it (RoleProvider, ThemeProvider,
// WorkspaceProvider, children) is identical. In `saas` the boundary is Clerk +
// ClerkApiClientProvider (the real token getter); in `local` it is LocalAuthProvider
// (a no-op token getter over the same seam), and NO Clerk symbol executes.
function AuthBoundary({ children }: { children: React.ReactNode }) {
  if (!isSaaS) {
    return <LocalAuthProvider>{children}</LocalAuthProvider>
  }
  return (
    <ClerkProvider
      signInFallbackRedirectUrl="/"
      signUpFallbackRedirectUrl="/"
      // Clerk theming — Studio-aligned. Studio is the locked rebrand direction,
      // so the Clerk surfaces (sign-in/sign-up/user menu/avatar ring) use the
      // Studio palette across all themes. Reads as brand-coherent in classic
      // mode too because Studio's primary (near-black) is a neutral.
      // PRD §1 semantic lock + audit cross-cutting A.
      appearance={{
        baseTheme: undefined,
        variables: {
          colorPrimary: '#1a1814',         // near-black — positive primary
          colorBackground: '#ffffff',      // white paper card
          colorText: '#1a1814',            // warm ink
          colorInputBackground: '#fbf8f1', // cream surface
          colorInputText: '#1a1814',
          colorTextSecondary: '#5a5448',   // muted ink
          colorDanger: '#c44a1a',          // burnt orange — consequence
          fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif',
          borderRadius: '0.5rem',          // Studio uses 6-10px corners
        },
        elements: {
          card: 'border border-border bg-card',
          headerTitle: 'text-foreground',
          headerSubtitle: 'text-muted-foreground',
          socialButtonsBlockButton:
            'bg-card hover:bg-secondary text-foreground border border-border rounded-md',
          formFieldInput:
            'bg-background border border-border text-foreground placeholder:text-muted-foreground rounded-md',
          formButtonPrimary:
            'bg-primary hover:bg-primary/90 text-primary-foreground rounded-md',
          footerActionLink: 'text-primary hover:text-primary/80',
        },
        layout: {
          logoImageUrl: '/brand/automatos-mark-hi.png',
        },
      }}
    >
      <ClerkApiClientProvider>{children}</ClerkApiClientProvider>
    </ClerkProvider>
  )
}

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
    <AuthBoundary>
      <RoleProvider>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          themes={['light', 'dark', 'matte', 'studio']}
          storageKey="automatos-theme"
          disableTransitionOnChange
        >
          <QueryClientProvider client={queryClient}>
            <WorkspaceProvider>
              <Suspense fallback={null}>
                <StudioThemeFlag />
              </Suspense>
              {/* PRD-222: a new workspace lands directly in the conversation
                  where Auto opens (OnboardingOpener) — no first-login modal or
                  guided tour (both retired in W2·S5). */}
              {children}
              <GlobalSearch />
              <Toaster position="top-right" richColors closeButton />
            </WorkspaceProvider>
          </QueryClientProvider>
        </ThemeProvider>
      </RoleProvider>
    </AuthBoundary>
  )
}
