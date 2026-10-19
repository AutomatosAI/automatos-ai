/**
 * PRD-261 A-10b — SDK area `navigation` (frontend/lib/theme-sdk/navigation.tsx): moving around the app without
 * importing `next/*`. Every href is checked: a path inside the app, or https out of it. Anything else throws a
 * TypeError, never a navigation.
 */
import type { AnchorHTMLAttributes, ForwardRefExoticComponent, PropsWithoutRef, ReactNode, RefAttributes } from 'react'

export interface CoreNavItem {
  readonly id: string
  readonly label: string
  /** One line about the page; null when the menu has none. */
  readonly description: string | null
  readonly href: string
  /** The menu group ('OPERATIONS', 'WORKFORCE', 'WORKSPACE'); null for the footer. */
  readonly group: string | null
  /** Opens in a new tab (Docs). */
  readonly external: boolean
}

export interface CoreNav {
  /** The menu's items in the menu's own order, groups included. */
  readonly primary: readonly CoreNavItem[]
  /** Docs and Settings, where Studio shows them at the foot of its sidebar. */
  readonly footer: readonly CoreNavItem[]
}

/** The App Router, taking only paths inside the app. */
export interface HostRouter {
  push(href: string): void
  replace(href: string): void
  back(): void
  refresh(): void
  prefetch(href: string): void
}

/** The query string, read-only: a theme reads its parameters and never edits the router's own. */
export type HostSearchParams = Pick<URLSearchParams, 'get' | 'getAll' | 'has' | 'toString' | 'forEach' | 'keys' | 'values' | 'entries'>

export interface HostLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href' | 'target' | 'rel'> {
  /** A path inside the app, or an https URL (opened in a new tab). */
  readonly href: string
  readonly children?: ReactNode
}

/** A question card, as questionHref reads it: the card, what it is about, and whose job it is. */
export interface HostQuestionRef {
  readonly id: number
  readonly subject_type: string
  readonly subject_id: string
  readonly owner?: {
    readonly agent?: { readonly id: number; readonly name?: string | null } | null
    readonly ticket?: { readonly id: number; readonly title?: string | null; readonly number?: string | null } | null
  } | null
}

/** The board tab of the command center. */
export declare const BOARD_HREF: string
/** The questions tab of the command center. */
export declare const QUESTIONS_HREF: string
export declare const CHAT_HREF: string
export declare const DELIVERABLES_HREF: string
/** The deliverables file explorer. */
export declare const DELIVERABLE_EXPLORER_HREF: string
/** The query parameter that opens one deliverable on the deliverables page. */
export declare const DELIVERABLE_PARAM: string
export declare const LOOKS_HREF: string
export declare const LOOK_STYLE_HREF: string

/** One ticket on the board, optionally with one of its questions open. */
export declare function ticketHref(taskId: string | number, questionId?: string | number | null): string
/** A question's ticket with the question open, or the questions tab when it has no ticket. */
export declare function questionHref(question: HostQuestionRef): string
export declare function missionHref(missionId: string | number): string
export declare function chatHref(chatId: string): string

/** A link: the host's client-side link for a path, a new tab for https, a TypeError for anything else. */
export declare const HostLink: ForwardRefExoticComponent<PropsWithoutRef<HostLinkProps> & RefAttributes<HTMLAnchorElement>>

export declare function useHostRouter(): HostRouter
/** The current path, without the query string. */
export declare function useHostPathname(): string
/** The current query string, as a copy that changes when the URL does. */
export declare function useHostSearchParams(): HostSearchParams
/** The core nav for this caller: the Studio menu after the edition, role and plan filters. */
export declare function useCoreNav(): CoreNav
/** The id of the core nav item `pathname` belongs to, or null. */
export declare function resolveActiveNavId(pathname: string): string | null
