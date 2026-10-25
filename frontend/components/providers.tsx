
'use client'

import { ClerkProvider } from '@clerk/nextjs'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { ThemeProvider } from './theme-provider'
import { WorkspaceProvider } from './workspace-provider'
import { ClerkApiClientProvider } from './clerk-api-client-provider'
import { LocalAuthProvider } from './local-auth-provider'
import { RoleProvider } from '../contexts/role-context'
import { UiStyleProvider } from '../contexts/ui-style-context'
import type { UiStyle } from '@/lib/ui-style'
import type { ThemeTone } from '@/lib/theme-host/boot'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import { Toaster } from './ui/sonner'
import { GlobalSearch } from './shared/global-search'
import { ChooseInstallProvider } from './settings/themes/choose-install'
import { ThemeHostProvider } from './theme-host/theme-host-provider'
import { ThemeLookRuntime } from './theme-host/theme-look-runtime'
import { isSaaS } from '@/lib/auth-edition'

// PRD-244 W0: the `?theme=studio-preview` URL flag is gone — Studio is a
// picker entry like any other theme (D1a); the default stays `system` until
// the Wave-4 pass is signed (D1b).

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

interface ProvidersProps {
  children: React.ReactNode
  /** PRD-244 D1: the Style axis, read from its cookie by app/layout.tsx. */
  initialUiStyle?: UiStyle
  /** PRD-260 K4, PRD-261 A-03: the boot's tone (a theme whose tone follows its look), forced from the first paint. */
  initialTone?: ThemeTone
  /** PRD-261 W-A6b: an installed theme's registry entry, from its boot (lib/theme-host/installed-entry.ts); null otherwise. */
  initialTheme?: ThemeEntry | null
}

export function Providers({ children, initialUiStyle, initialTone, initialTheme = null }: ProvidersProps) {
  // PRD-260 K4, PRD-261 A-13c: under a theme whose tone follows its look, the look's ground
  // sets the tone (the look runtime hands it up). forcedTheme forces it without overwriting
  // the user's stored tone, which returns when they leave the theme; it starts from the
  // server's tone cookie, so next-themes' script forces it before paint. Forced at every
  // width: the look runtime paints <html> below the theme's minWidth too, and Studio's
  // semantic tokens and dark: classes must match its ground there.
  const [lookTone, setLookTone] = useState<ThemeTone | undefined>(initialTone)
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
          defaultTheme="dark"
          enableSystem
          themes={['light', 'dark']}
          storageKey="automatos-theme"
          disableTransitionOnChange
          forcedTheme={lookTone}
        >
          <UiStyleProvider initialStyle={initialUiStyle} installedTheme={initialTheme}>
          <QueryClientProvider client={queryClient}>
            <WorkspaceProvider>
              {/* PRD-261 A-07d: one chooser for installed themes, around every page (Settings, the
                  Marketplace, the header's theme menu) and ⌘K, so a private version's opt-in
                  dialog opens from any of them before the choice is stored. */}
              <ChooseInstallProvider>
                {/* PRD-222: a new workspace lands directly in the conversation
                    where Auto opens (OnboardingOpener) — no first-login modal or
                    guided tour (both retired in W2·S5). */}
                <ThemeHostProvider>
                  {/* PRD-261 A-13c: the active theme's look runtime, at every width, switched on only
                      while the theme declares looks (Classic and Studio get none); it stays mounted
                      either way, so a style switched in place never remounts the page (M69).
                      PRD-260 F5: looks are the signed-in user's in this workspace, so it sits inside
                      the WorkspaceProvider. */}
                  <ThemeLookRuntime onToneChange={setLookTone}>{children}</ThemeLookRuntime>
                </ThemeHostProvider>
                <GlobalSearch />
              </ChooseInstallProvider>
              <Toaster position="top-right" richColors closeButton />
            </WorkspaceProvider>
          </QueryClientProvider>
          </UiStyleProvider>
        </ThemeProvider>
      </RoleProvider>
    </AuthBoundary>
  )
}
