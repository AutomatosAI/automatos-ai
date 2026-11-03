'use client'

/**
 * PRD-268 §6.2.4 — who is signed in, as the server resolves them: GET /api/profile (PRD-233 S6, both editions),
 * whose `id` is the person's users.id, the id a suggestion's `user_id` names; null when the server has no row for them.
 *
 * The Theme SDK's useSuggestions reads it to hand a theme only the signed-in person's own suggestions (the open list
 * gives an owner or admin the whole workspace's). The answer is kept per session identity (the identity provider's
 * user id; the local edition has one operator and none): a sign-out and another person's sign-in in the same tab ask
 * again, and never read the previous person's answer. Nothing is asked until the session has loaded and is signed in.
 */
import { useQuery } from '@tanstack/react-query'
import { apiClient, type ProfileResponse } from '@/lib/api-client'
import { useAuth } from '@/lib/auth-hooks'

export const profileQueryKeys = {
  me: (sessionUserId: string | null) => ['profile', 'me', sessionUserId] as const,
}

/** The signed-in person's profile. */
export function useProfile() {
  const { isLoaded, isSignedIn, userId } = useAuth()
  return useQuery<ProfileResponse>({
    queryKey: profileQueryKeys.me(userId ?? null),
    queryFn: () => apiClient.getProfile(),
    enabled: Boolean(isLoaded && isSignedIn),
    staleTime: Infinity,
  })
}
