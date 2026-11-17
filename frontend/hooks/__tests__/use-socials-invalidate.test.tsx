/**
 * One calendar, two views (16 Nov 2026): the Command Center calendar shows the workspace's
 * coming social posts, so a change to a post in Socials (a move, an approval, a delete) must
 * be read again there too. `useInvalidateSocials` marks both the Socials queries and the
 * Command Center's stale; before, only a schedule write did, so a post moved in the Socials
 * calendar stayed at its old slot in the Command Center until its own timer came round.
 */
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/lib/api-client', () => ({ apiClient: {}, default: {} }))

import { activityQueryKeys } from '@/hooks/use-activity-api'
import { socialsQueryKeys, useInvalidateSocials } from '@/hooks/use-socials-api'

describe('after a social post changed', () => {
  it('the Socials queries and the Command Center calendar feed are both read again', async () => {
    const client = new QueryClient()
    client.setQueryData(socialsQueryKeys.posts('w1'), { posts: [] })
    client.setQueryData(activityQueryKeys.schedule('7d'), { scheduled: [] })
    client.setQueryData(activityQueryKeys.schedule('30d'), { scheduled: [] })
    client.setQueryData(['agents', 'list'], [])
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>

    const { result } = renderHook(() => useInvalidateSocials(), { wrapper })
    await result.current()

    const stale = (key: readonly unknown[]) => client.getQueryState(key)?.isInvalidated
    expect(stale(socialsQueryKeys.posts('w1'))).toBe(true)
    expect(stale(activityQueryKeys.schedule('7d'))).toBe(true)
    expect(stale(activityQueryKeys.schedule('30d'))).toBe(true)
    expect(stale(['agents', 'list'])).toBe(false) // nothing else is touched
  })
})
