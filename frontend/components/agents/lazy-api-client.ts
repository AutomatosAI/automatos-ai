/**
 * #942 — the one lazy import of `apiClient` for the agent form's runtime hooks.
 *
 * The Runtime section's hooks each did `await import('@/lib/api-client')` on mount, in the same
 * tick. In vitest (2.1) two concurrent dynamic imports of a `vi.doMock`ed module from ONE module
 * race: the importer's callstack is shared, the first import marks the mock as in progress on it,
 * and the second falls through to the real client. CI showed it: the settings call
 * (`/api/v1/cli-hosts/settings`) reached the real client in every test while the health call got
 * the mock, so the session tool line never rendered in a test. The page itself was unaffected.
 * One shared promise means one import, whoever asks first; a failed import is not cached.
 */

import type { apiClient as ApiClient } from '@/lib/api-client'

let pending: Promise<typeof ApiClient> | null = null

/** The app's API client, imported once on first use. */
export function loadApiClient(): Promise<typeof ApiClient> {
  if (!pending) {
    pending = import('@/lib/api-client').then(
      (mod) => mod.apiClient,
      (error: unknown) => {
        pending = null
        throw error
      },
    )
  }
  return pending
}
