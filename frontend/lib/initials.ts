/**
 * A person's initials for an avatar with no picture: the profile button (local), the local profile
 * form and the enterprise profile. Its own module, with no Clerk import, so every edition can read it.
 */

/** "Ada Lovelace" → "AL"; one word → its first two letters; else the email's first letter. */
export function initialsFor(name: string | null | undefined, email: string | null | undefined): string {
  const parts = (name ?? '').trim().split(/\s+/).filter(Boolean)
  if (parts.length >= 2) return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase()
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  const fromEmail = (email ?? '').trim().charAt(0)
  return fromEmail ? fromEmail.toUpperCase() : '?'
}
