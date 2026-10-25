/**
 * PRD-261 A-07d — shared rows and a render helper for the Settings › Appearance › Themes tests.
 * The shapes are the wire types of lib/api-client.ts (the answers of orchestrator/api/themes_*.py).
 */
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

import type { InstalledTheme, ThemeSettings } from '@/lib/api-client'

import { ChooseInstallProvider } from '../choose-install'

// services/themes/optin.py OPT_IN_WORDING: the dialog shows the server's sentence.
export const OPT_IN_WORDING = 'This theme was not reviewed. Its code runs with your access to all your workspaces.'

export const INSTALL_ID = '11111111-1111-4111-8111-111111111111'
export const PACKAGE_ID = '22222222-2222-4222-8222-222222222222'
export const NEXT_PACKAGE_ID = '33333333-3333-4333-8333-333333333333'

// step_up.py step_up_body(10).
export const STEP_UP_BODY = {
  clerk_error: {
    type: 'forbidden',
    reason: 'reverification-error',
    metadata: { reverification: { level: 'first_factor', afterMinutes: 10 } },
  },
  code: 'step_up_required',
  detail: "Confirm it's you to continue: this action needs a sign-in from the last 10 minutes.",
}

/** The error apiClient throws for the step-up 403: the parsed body rides it (§2.2). */
export function stepUpError(): Error {
  return Object.assign(new Error(STEP_UP_BODY.detail), { status: 403, body: STEP_UP_BODY })
}

export function installRow(overrides: Partial<InstalledTheme> = {}): InstalledTheme {
  return {
    installId: INSTALL_ID,
    themeId: 'acme/hello',
    name: 'Hello theme',
    packageId: PACKAGE_ID,
    version: '1.0.0',
    ownerType: 'workspace',
    reviewed: false,
    private: true,
    isDefault: false,
    pinnedVersion: null,
    previousPackageId: null,
    pendingPackageId: null,
    pendingVersion: null,
    installedAt: '2026-10-19T10:00:00+00:00',
    optIn: { required: true, wording: OPT_IN_WORDING, packageId: PACKAGE_ID },
    revoked: null,
    needsUpdate: false,
    signature: { state: 'unsigned', keyId: null },
    ...overrides,
  }
}

export function themeSettings(overrides: Partial<ThemeSettings> = {}): ThemeSettings {
  return {
    privateUploads: { setting: null, source: 'default', default: 'on', platform: 'auto', effective: 'on' },
    membersMayChoose: true,
    installRole: 'admins',
    installRoles: ['admins', 'owner'],
    canInstall: true,
    marketplace: 'on',
    defaultInstallId: null,
    pins: [],
    updatedAt: null,
    ...overrides,
  }
}

/**
 * `ui` under a query client and the installed-theme chooser, as components/providers.tsx mounts them
 * above every page: "Use this theme" chooses, and opens the opt-in dialog, through the chooser.
 */
export function renderWithQuery(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ChooseInstallProvider>{ui}</ChooseInstallProvider>
    </QueryClientProvider>,
  )
}
