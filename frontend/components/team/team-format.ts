/**
 * Display helpers shared by the team page's header and member rows. Moved out of
 * team-management.tsx unchanged.
 */
import type { TeamMember } from './team-types'

/** "Active this week": the pill's count and the row highlight share this window. */
export const ACTIVE_WITHIN_DAYS = 7

/** "1 chat" / "4 chats" — the activity cell reads as a sentence, not a tally. */
export function plural(n: number, noun: string): string {
    return `${n} ${noun}${n === 1 ? '' : 's'}`
}

/**
 * The display label when `users.name` is empty — the email's local part.
 * Every Clerk-provisioned user has a NULL name today (names were never synced
 * into `users`), so "Unknown User" was the whole roster. This is derived from
 * data the row already carries, not invented.
 */
export function displayName(member: Pick<TeamMember, 'name' | 'email'>): string {
    return member.name || member.email.split('@')[0]
}

/** The role select's text, border and fill classes for each workspace role. */
export function getRoleBadgeColor(role: string) {
    switch (role) {
        case 'owner': return 'text-purple-400 border-purple-400/30 bg-purple-400/10'
        case 'admin': return 'text-warning border-warning/30 bg-warning/10'
        case 'editor': return 'text-blue-400 border-blue-400/30 bg-blue-400/10'
        case 'viewer': return 'text-green-400 border-green-400/30 bg-green-400/10'
        default: return 'text-muted-foreground border-border bg-secondary/50'
    }
}
