'use client'

/**
 * PRD-265 US-011 — Settings' open tab lives in the address, so a link can open a tab.
 *
 * `/settings?tab=billing` opens Billing: the paused banner's "Top up" (TOP_UP_HREF), Auto's
 * onboarding link (`/settings?tab=billing&plan=pro`) and Stripe's return pages all land there.
 * Choosing a tab switches at once and rewrites `?tab=` with `router.replace`, keeping the
 * other parameters, so tab clicks never pile up in the browser history. A tab the address
 * names that this person does not have (unknown, hosted-only in the local edition, admin-only)
 * opens the default instead.
 */
import { useCallback, useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

export const SETTINGS_HREF = '/settings'
export const SETTINGS_TAB_PARAM = 'tab'

type SettingsHref = typeof SETTINGS_HREF | `${typeof SETTINGS_HREF}?${string}`

/** The Settings address with these parameters (none: plain `/settings`). */
export function settingsHref(params: URLSearchParams): SettingsHref {
  const query = params.toString()
  return query ? `${SETTINGS_HREF}?${query}` : SETTINGS_HREF
}

/** `requested` when it is one of `tabs`, else `fallback`. */
export function resolveSettingsTab(requested: string | null, tabs: readonly { value: string }[], fallback: string): string {
  return requested !== null && tabs.some((tab) => tab.value === requested) ? requested : fallback
}

/** The open tab and the way to open another; `fallback` is the tab when the address names none. */
export function useSettingsTab(tabs: readonly { value: string }[], fallback: string): readonly [string, (next: string) => void] {
  const router = useRouter()
  const searchParams = useSearchParams()
  const requested = searchParams?.get(SETTINGS_TAB_PARAM) ?? null
  // The tab just clicked, shown before the address catches up; a new address wins over it.
  const [picked, setPicked] = useState<string | null>(null)
  useEffect(() => setPicked(null), [requested])

  const select = useCallback(
    (next: string) => {
      setPicked(next)
      const params = new URLSearchParams(searchParams?.toString() ?? '')
      params.set(SETTINGS_TAB_PARAM, next)
      router.replace(settingsHref(params), { scroll: false })
    },
    [router, searchParams],
  )

  return [resolveSettingsTab(picked ?? requested, tabs, fallback), select] as const
}
