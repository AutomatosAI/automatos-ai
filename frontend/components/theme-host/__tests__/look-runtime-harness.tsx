/**
 * PRD-261 A-19f2 — the providers as components/providers.tsx nests them around the host's look
 * runtime, for its tests: next-themes' ThemeProvider (forcedTheme takes the tone the look forces),
 * the style context with the installed theme's entry (as the root layout builds it from the boot),
 * the query client, the theme host's status and ThemeLookRuntime. The installed theme with looks is
 * hello-a09b (lib/theme-host/looks/__tests__/hello-looks.ts). A test mocks the account
 * (@/lib/auth-hooks) and the workspace (@/components/workspace-provider) itself.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, type ReactNode } from 'react'

import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { ThemeLookRuntime } from '@/components/theme-host/theme-look-runtime'
import { ThemeProvider } from '@/components/theme-provider'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import type { ThemeTone } from '@/lib/theme-host/boot'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { HELLO_ENTRY } from '@/lib/theme-host/looks/__tests__/hello-looks'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import type { UiStyle } from '@/lib/ui-style'

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
export const TONE_STORAGE_KEY = 'automatos-theme'
/** The user's own tone before any look forces one (components/providers.tsx defaultTheme). */
export const USER_DEFAULT_TONE: ThemeTone = 'dark'

/** The install of hello-a09b, a small theme with looks whose tone follows its look. */
export const HELLO_LOOKS_STYLE: UiStyle = 'theme:install:6e1f3b8f-2d5c-4fab-a4e7-9a2b3d4f5c6e'

function entryOf(style: UiStyle): ThemeEntry {
  const entry = installedThemeEntry(style, HELLO_ENTRY)
  if (entry === null) throw new Error(`${style} names no installed theme`)
  return entry
}

/** hello-a09b's registry entry, as the root layout builds it from its boot (lib/theme-host/installed-entry.ts). */
export const HELLO_LOOKS_THEME: ThemeEntry = entryOf(HELLO_LOOKS_STYLE)

export function hostValue(status: ThemeHostStatus): ThemeHostContextValue {
  return { status, reason: status === 'failed' ? 'load' : null, frame: null, slots: null, reportRenderFailure: () => undefined }
}

/** A fresh cache, as components/providers.tsx has one per page; no retries, no logging. */
export function quietQueryClient(): QueryClient {
  const quiet = () => undefined
  return new QueryClient({ defaultOptions: { queries: { retry: false } }, logger: { log: quiet, warn: quiet, error: quiet } })
}

export interface LookAppearanceProps {
  readonly style: UiStyle
  readonly status: ThemeHostStatus
  /** The installed theme's entry; none by default. Read once, at mount, as from the root layout. */
  readonly entry?: ThemeEntry | null
  /** A test that answers GET /health itself hands in the cache; otherwise one of its own. */
  readonly queryClient?: QueryClient
  readonly children: ReactNode
}

export function LookAppearance({ style, status, entry = null, queryClient, children }: LookAppearanceProps) {
  const [lookTone, setLookTone] = useState<ThemeTone | undefined>(undefined)
  const [ownClient] = useState(quietQueryClient)
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme={USER_DEFAULT_TONE}
      enableSystem
      themes={['light', 'dark']}
      storageKey={TONE_STORAGE_KEY}
      disableTransitionOnChange
      forcedTheme={lookTone}
    >
      <UiStyleProvider initialStyle={style} installedTheme={entry}>
        <QueryClientProvider client={queryClient ?? ownClient}>
          <ThemeHostContext.Provider value={hostValue(status)}>
            <ThemeLookRuntime onToneChange={setLookTone}>{children}</ThemeLookRuntime>
          </ThemeHostContext.Provider>
        </QueryClientProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}
