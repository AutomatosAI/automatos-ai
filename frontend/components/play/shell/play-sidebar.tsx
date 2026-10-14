'use client'

import { useId, useMemo } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ExternalLink } from 'lucide-react'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useSystemRoleOptional } from '@/contexts/role-context'
import { useBackendHealth } from '@/hooks/use-backend-health'
import { useRecentChats } from '@/hooks/use-recent-chats'
import { useStartChat } from '@/hooks/use-start-chat'
import { CHAT_HREF, chatHref } from '@/lib/ticket-links'
import { useChatSessionStore } from '@/stores/chat-session-store'

import { ConnectionPill } from './connection-pill'
import { buildPlayNav, resolvePlayActiveId, type PlayNav, type PlayNavGroup, type PlayNavItem } from './play-nav'
import styles from './play-sidebar.module.css'

/**
 * PRD-260 T-004 — Play's sidebar (Ember mock 01), ported from Automatos Play's
 * components/shell/Sidebar.tsx onto the core: the wordmark, New chat (a draft in
 * the chat session store, then /chat), every core page grouped as the Studio menu
 * has them (play-nav.ts), Recent (getChatHistory) and the connection pill (GET
 * /health). Rendered by PlayFrame, only under Play at >= 1024 px.
 */

/** A conversation the backend has not named yet. */
const UNTITLED_CHAT = 'Conversation'

/** The nav for this edition, role and plan — the filters StudioSidebar applies. */
function usePlayNav(): PlayNav {
  const exposure = useWorkspaceOptional()?.workspace?.exposure
  // Unknown (no provider) is not an admin: admin-only items are hidden, never guessed in.
  const isAdmin = useSystemRoleOptional()?.isAdmin ?? false
  return useMemo(() => buildPlayNav({ isAdmin, exposure }), [isAdmin, exposure])
}

export function PlaySidebar() {
  const pathname = usePathname() ?? ''
  const nav = usePlayNav()
  const activeId = resolvePlayActiveId(pathname)
  const backend = useBackendHealth()
  return (
    <aside className={styles.sidebar} aria-label="Sidebar">
      <Link href={CHAT_HREF} className={styles.wordmark}>
        <span>
          Automatos <span className={styles.wordmarkAccent}>Play</span>
        </span>
      </Link>
      <NewChatButton />
      <nav aria-label="Main" className={styles.navGroups}>
        {nav.groups.map((group) => (
          <NavGroupList key={group.id} group={group} activeId={activeId} />
        ))}
      </nav>
      <RecentList onChat={pathname === CHAT_HREF} />
      <div className={styles.foot}>
        {nav.footer.length > 0 && (
          <ul className={styles.list} aria-label="More">
            {nav.footer.map((item) => (
              <li key={item.id}>
                <NavLink item={item} active={activeId === item.id} />
              </li>
            ))}
          </ul>
        )}
        <ConnectionPill connection={backend.connection} host={backend.host} />
      </div>
    </aside>
  )
}

/** New chat: a fresh draft in the chat session store, then the chat. */
function NewChatButton() {
  const startChat = useStartChat()
  return (
    <button type="button" className={styles.newChat} onClick={startChat}>
      New chat
    </button>
  )
}

function NavLink({ item, active }: { readonly item: PlayNavItem; readonly active: boolean }) {
  if (item.external) {
    return (
      <a href={item.href} target="_blank" rel="noopener noreferrer" className={styles.navLink}>
        {item.label}
        <ExternalLink className={styles.externalIcon} aria-hidden="true" />
      </a>
    )
  }
  return (
    <Link href={item.href} className={styles.navLink} aria-current={active ? 'page' : undefined}>
      {item.label}
    </Link>
  )
}

/** A group: its small caps label over its links; Look has no label and stands alone. */
function NavGroupList({ group, activeId }: { readonly group: PlayNavGroup; readonly activeId: string | null }) {
  const labelId = useId()
  const list = (
    <ul className={styles.list}>
      {group.items.map((item) => (
        <li key={item.id}>
          <NavLink item={item} active={activeId === item.id} />
        </li>
      ))}
    </ul>
  )
  if (group.label === null) return list
  return (
    <div role="group" aria-labelledby={labelId}>
      <span id={labelId} className={styles.groupLabel}>
        {group.label}
      </span>
      {list}
    </div>
  )
}

/** Recent: the latest conversations, each opening in the chat. Nothing until there is one. */
function RecentList({ onChat }: { readonly onChat: boolean }) {
  const headingId = useId()
  const recent = useRecentChats()
  const activeChatId = useChatSessionStore((s) => s.session.activeChatId)
  if (recent.length === 0) return null
  return (
    <section aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.sectionLabel}>
        Recent
      </h2>
      <ul className={styles.list}>
        {recent.map((chat) => (
          <li key={chat.id}>
            <Link
              href={chatHref(chat.id)}
              className={styles.recentLink}
              aria-current={onChat && chat.id === activeChatId ? 'true' : undefined}
            >
              {chat.title || UNTITLED_CHAT}
            </Link>
          </li>
        ))}
      </ul>
    </section>
  )
}
