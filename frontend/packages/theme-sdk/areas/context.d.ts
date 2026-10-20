/**
 * PRD-261 A-10b — SDK area `context` (frontend/lib/theme-sdk/context.ts): what a theme may know about where it is
 * drawn. Each hook hands back a small, frozen, theme-neutral copy of the host's state, and answers "unknown"
 * (null, or the host chrome's own fail-open default) outside the host's providers; none throws.
 */

/** 'local' (one operator, no sign-in) or 'saas'. */
export type HostEdition = 'local' | 'saas'
export type HostWorkspaceRole = 'owner' | 'admin' | 'editor' | 'viewer' | 'member'
export type HostSystemRole = 'super_admin' | 'admin' | 'user'
export type HostTone = 'light' | 'dark'

export interface HostHealth {
  readonly connection: 'checking' | 'connected' | 'down'
  /** The backend's host name, for "Connected · <host>"; absent when it can't be read. */
  readonly host?: string
  /** An ask (or one of its retries) is in flight. */
  readonly checking: boolean
  /** Ask again now. */
  readonly check: () => void
}

export interface HostHealthOptions {
  /** Make this caller the page's one poller (the asleep screen); every other caller reads the same answer. */
  readonly poll?: boolean
}

/** The workspace the page is in: who it is, its plan, and whether Socials is offered in it. */
export interface HostWorkspace {
  readonly id: string
  readonly name: string
  readonly slug: string
  readonly plan: 'basic' | 'pro' | 'business' | 'enterprise'
  /** The platform offers Socials to this workspace, as the host's Deliverables page reads it; false while unknown. */
  readonly socialsAvailable: boolean
}

export interface HostWorkspaceState {
  /** Null while it loads, when nobody is signed in, or outside the host's provider. */
  readonly workspace: HostWorkspace | null
  readonly isLoading: boolean
}

export interface HostRole {
  /** The caller's role in this workspace; null until the workspace answers. */
  readonly workspaceRole: HostWorkspaceRole | null
  /** The caller's platform role; null until the session answers. */
  readonly systemRole: HostSystemRole | null
  /** A platform admin. Unknown is not an admin. */
  readonly isAdmin: boolean
  /** False for a workspace viewer: write affordances render disabled. Open while unknown. */
  readonly canEdit: boolean
}

export interface HostTrial {
  readonly state: 'active' | 'warned' | 'exhausted' | 'converted'
  readonly grantedUsd: number
  readonly spentUsd: number
}

export interface HostMediaPreferences {
  /** The user asks for reduced motion (always true on the server, so a first paint never animates). */
  readonly reducedMotion: boolean
}

/** The workspace the page is in, as a theme may see it. */
export declare function useWorkspace(): HostWorkspaceState

/** The caller's workspace and platform roles. */
export declare function useRole(): HostRole

/** The edition; fixed for the life of the build. */
export declare function useEdition(): HostEdition

/** The workspace's trial ledger; null when it has none or has not loaded. */
export declare function useTrial(): HostTrial | null

/** Whether the backend answers. */
export declare function useHealth(options?: HostHealthOptions): HostHealth

/** Whether the browser is online. */
export declare function useOnline(): boolean

/** The user's media preferences. */
export declare function useMediaPreferences(): HostMediaPreferences

/** The tone the page is painted in, forced or chosen; null before the host knows it (the server). */
export declare function useTone(): HostTone | null
