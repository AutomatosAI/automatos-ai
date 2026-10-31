/**
 * PRD-261 A-08c — the browser's way to an installed theme's boot when the page's carries no entry.
 *
 * The root layout asks GET /api/themes/boot/{installId} on the server within 300 ms
 * (boot-server.ts). When that fetch times out, is refused (a local install in
 * LOCAL_REMOTE_ACCESS=paired mode answers the server 401: the layout holds no device token) or
 * fails, the page carries the base boot: no entry, no prepaint table, no presets. The client host
 * still loads the theme from GET /api/themes/active, but with no entry there is no active theme and
 * no look runtime, so a theme whose colours all come from its prepaint table draws with no look.
 *
 * So the browser asks the boot route itself, through apiClient (the Clerk session's bearer, or the
 * paired browser's device token) and with no 300 ms cap, and hands the answer to the two places the
 * page's boot feeds: the active theme's entry (contexts/ui-style-context.tsx) and the look
 * runtime's rules (lib/theme-host/looks/provider.tsx). The first paint of the look comes late, but
 * it comes. The answer passes the same gate as the server's (parseInstalledBoot, boot-sanitize.ts).
 *
 * One request per install for the page's life: both places ask the same source and get the same
 * answer, and a failure is not asked again until the next load (it is logged once, and the theme
 * runs with no look, as before this fallback).
 */
import { apiClient } from '@/lib/api-client'

import { parseInstalledBoot } from './boot-sanitize'
import type { ThemeBoot } from './boot-types'

const LOG_PREFIX = '[theme-boot]'

/** An install's boot as the browser may use it; null when it could not get one. Never rejects. */
export type InstalledBootSource = (installId: string) => Promise<ThemeBoot | null>

/** What the boot route answered for an install, unchecked. */
export type InstalledBootFetch = (installId: string) => Promise<unknown>

/** One request for the install's boot, through the boot gate; null, with one warning, when it fails. */
async function fetchCheckedBoot(installId: string, fetchBoot: InstalledBootFetch): Promise<ThemeBoot | null> {
  try {
    return parseInstalledBoot(await fetchBoot(installId))
  } catch (error) {
    console.warn(LOG_PREFIX, "the browser couldn't fetch the installed theme's boot either, so it runs with no look", error)
    return null
  }
}

/** A source that asks once per install and hands every later caller that same answer. */
export function createInstalledBootSource(fetchBoot: InstalledBootFetch): InstalledBootSource {
  const asked = new Map<string, Promise<ThemeBoot | null>>()
  return (installId) => {
    const held = asked.get(installId)
    if (held !== undefined) return held
    const answer = fetchCheckedBoot(installId, fetchBoot)
    asked.set(installId, answer)
    return answer
  }
}

/** The page's source: GET /api/themes/boot/{installId} through apiClient, once per install. */
export const pageInstalledBoot: InstalledBootSource = createInstalledBootSource((installId) => apiClient.getThemeBoot(installId))
