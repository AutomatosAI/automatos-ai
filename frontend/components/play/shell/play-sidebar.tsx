'use client'

import { useId, useMemo } from 'react'
import { ExternalLink } from 'lucide-react'
import {
  CHAT_HREF,
  HostLink,
  chatHref,
  useCoreNav,
  useHealth,
  useHostPathname,
  useRecentChats,
  useStartChat,
} from '@automatos/theme-sdk'

import { ConnectionPill } from './connection-pill'
import { buildPlayNav, resolvePlayActiveId, type PlayNav, type PlayNavGroup, type PlayNavItem } from './play-nav'
import styles from './play-sidebar.module.css'

/**
 * PRD-260 T-004 — Play's sidebar (Ember mock 01), ported from Automatos Play's
 * components/shell/Sidebar.tsx onto the core: the wordmark, New chat (a draft in
 * the chat session store, then /chat), every core page grouped as the Studio menu
 * has them (play-nav.ts), Recent (getChatHistory) and the connection pill (GET
 * /health). Rendered by PlayFrame, only under Play at >= 1024 px. PRD-261 A-14f:
 * each of those reaches the core through '@automatos/theme-sdk' (useStartChat,
 * useCoreNav, useRecentChats, useHealth, HostLink).
 */

/** A conversation the backend has not named yet. */
const UNTITLED_CHAT = 'Conversation'

/** The nav for this edition, role and plan: the core nav (StudioSidebar's filters), Play's pages added. */
function usePlayNav(): PlayNav {
  // Unknown (no provider) is not an admin: the core nav hides admin-only items, never guesses them in.
  const coreNav = useCoreNav()
  return useMemo(() => buildPlayNav(coreNav), [coreNav])
}

export function PlaySidebar() {
  const pathname = useHostPathname()
  const nav = usePlayNav()
  const activeId = resolvePlayActiveId(pathname)
  const backend = useHealth()
  return (
    <aside className={styles.sidebar} aria-label="Sidebar">
      <HostLink href={CHAT_HREF} className={styles.wordmark}>
        <span>
          Automatos <span className={styles.wordmarkAccent}>Play</span>
        </span>
      </HostLink>
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

/** New chat: a fresh draft in the chat session store, then the chat; pressable once the store is hydrated. */
function NewChatButton() {
  const { startChat, ready } = useStartChat()
  return (
    <button type="button" className={styles.newChat} onClick={startChat} disabled={!ready}>
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
    <HostLink href={item.href} className={styles.navLink} aria-current={active ? 'page' : undefined}>
      {item.label}
    </HostLink>
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
  const { chats: recent, activeChatId } = useRecentChats()
  if (recent.length === 0) return null
  return (
    <section aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.sectionLabel}>
        Recent
      </h2>
      <ul className={styles.list}>
        {recent.map((chat) => (
          <li key={chat.id}>
            <HostLink
              href={chatHref(chat.id)}
              className={styles.recentLink}
              aria-current={onChat && chat.id === activeChatId ? 'true' : undefined}
            >
              {chat.title || UNTITLED_CHAT}
            </HostLink>
          </li>
        ))}
      </ul>
    </section>
  )
}
