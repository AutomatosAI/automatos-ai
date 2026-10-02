/**
 * PRD-253 S3.1 — what a paired host says about each CLI it knows: its name and
 * version, then who it runs as (GitHub Copilot: the account whose seat is spent)
 * or why the host does not serve it. Pure, for the host lines in Settings.
 */
import { sessionCli } from '@/lib/session-clis'

export interface HostCliInfo {
  version?: string | null
  path?: string | null
  served?: boolean
  reason?: string | null
  /** GitHub Copilot: `<login>@<host>` — never a credential. */
  login?: string | null
  /** GitHub Copilot: `keychain` or `gh` — where that login is read. */
  login_route?: string | null
}

export interface HostCliLine {
  id: string
  served: boolean
  text: string
}

const LOGIN_ROUTES: Record<string, string> = { keychain: 'keychain login', gh: 'GitHub CLI login' }

export function hostCliLines(clis: Record<string, HostCliInfo> | null | undefined): HostCliLine[] {
  return Object.entries(clis ?? {}).map(([id, info]) => {
    const label = sessionCli(id).label
    const name = info?.version ? `${label} · ${info.version}` : label
    if (!info?.served) return { id, served: false, text: `${name} — ${info?.reason || 'not served'}` }
    const route = info.login_route ? ` (${LOGIN_ROUTES[info.login_route] ?? info.login_route})` : ''
    return { id, served: true, text: info.login ? `${name} — runs as ${info.login}${route}` : name }
  })
}
