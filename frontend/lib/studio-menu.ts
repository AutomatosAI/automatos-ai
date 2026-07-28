/**
 * Studio sidebar menu — single source of truth.
 *
 * Locked from the round-3 shell delivery (`DUMPING AREA/sidebar-header-shell/
 * menu.jsx`). 13 primary items in 3 groups + 2 footer items. British spelling.
 * Routes match the live Next app routes. Lucide icon names.
 *
 * PRD §1 / shell rollout reference.
 */

import type { LucideIcon } from 'lucide-react';
import {
  MessagesSquare,
  LayoutDashboard,
  ClipboardList,
  Package,
  Bot,
  PlugZap,
  Library,
  Store,
  Users,
  TrendingUp,
  Building2,
  BookOpen,
  Settings,
} from 'lucide-react';

export type StudioMenuGroup = 'OPERATIONS' | 'WORKFORCE' | 'WORKSPACE';

export interface StudioMenuItem {
  /** Stable id for active-state matching */
  id: string;
  /** Visible label */
  label: string;
  /** One-line description (shown in icon-rail tooltip, optionally below the label) */
  desc: string;
  /** Next route */
  href: string;
  /** Lucide icon */
  icon: LucideIcon;
  /** Group bucket */
  group: StudioMenuGroup;
}

export interface StudioFooterItem {
  id: string;
  label: string;
  desc?: string;
  href: string;
  icon: LucideIcon;
  /** If true, opens in new tab + shows external-link affordance */
  external?: boolean;
}

export const STUDIO_MENU_PRIMARY: StudioMenuItem[] = [
  // OPERATIONS — daily-use surfaces
  { id: 'chat',     label: 'Chat',             desc: 'Your AI workspace',                  href: '/chat',           icon: MessagesSquare,    group: 'OPERATIONS' },
  { id: 'cmd',      label: 'Command Centre',   desc: 'Your AI workforce at a glance',      href: '/command-center', icon: LayoutDashboard,   group: 'OPERATIONS' },
  { id: 'assign',   label: 'Assignments',      desc: 'Plan, schedule & orchestrate',       href: '/assignments',    icon: ClipboardList,     group: 'OPERATIONS' },
  { id: 'deliv',    label: 'Deliverables',     desc: 'Files, code & agent output',         href: '/deliverables',   icon: Package,           group: 'OPERATIONS' },

  // WORKFORCE — managing agents + capabilities
  { id: 'agents',   label: 'Agent Management', desc: 'Manage AI agents and skills',        href: '/agents',         icon: Bot,               group: 'WORKFORCE' },
  { id: 'tools',    label: 'Tools & Integrations', desc: 'Development and utility tools',  href: '/tools',          icon: PlugZap,           group: 'WORKFORCE' },
  { id: 'kb',       label: 'Knowledge Base',   desc: 'Documents, databases & code',        href: '/documents',      icon: Library,           group: 'WORKFORCE' },
  { id: 'market',   label: 'Marketplace',      desc: 'Discover agents, playbooks',         href: '/marketplace',    icon: Store,             group: 'WORKFORCE' },

  // WORKSPACE — admin + decision economics
  { id: 'team',     label: 'Team Management',  desc: 'Manage workspace members',           href: '/team',           icon: Users,             group: 'WORKSPACE' },
  { id: 'analytics',label: 'Analytics',        desc: 'Performance, costs & insights',      href: '/analytics',      icon: TrendingUp,        group: 'WORKSPACE' },
  { id: 'admin',    label: 'Workspace Admin',  desc: 'Manage all workspaces',              href: '/admin/workspaces', icon: Building2,       group: 'WORKSPACE' },
];

export const STUDIO_MENU_FOOTER: StudioFooterItem[] = [
  { id: 'docs',     label: 'Docs',     href: 'https://docs.automatos.app', icon: BookOpen, external: true },
  { id: 'settings', label: 'Settings', desc: 'Profile, API keys, preferences', href: '/settings', icon: Settings },
];

/**
 * Per-page sub-nav tabs rendered by the generic <StudioPageTabs /> under the
 * header. Pages with bespoke composed layouts (Command Centre, Chat) render
 * their own tabs as part of their editorial page frame and are deliberately
 * absent from this map.
 *
 * Labels only — no seed counts. The previous placeholder badges (All 18,
 * Outputs 41, Skills 24…) fabricated numbers pilots read as real (PRD-154 S10).
 * Pages with honest counts wire their own dynamic numbers.
 */
/**
 * Tuple shape: [label, count, hrefOverride?]
 * hrefOverride is used when a tab needs to navigate to a different route
 * instead of `?tab=<slug>` on the menu's base href (e.g. Deliverables →
 * Explorer is its own page, not a tab panel).
 */
export const STUDIO_PAGE_TABS: Record<string, Array<[string, number, string?]>> = {
  assign:  [['All', 18], ['Mine', 4], ['Scheduled', 9], ['Drafts', 2]],
  deliv:   [['Outputs', 41], ['Blogs', 6], ['Templates', 12], ['Explorer', 0, '/deliverables/explorer']],
  agents:  [['Roster', 11], ['Skills', 24], ['Lineage', 0], ['Settings', 0]],
};

/**
 * Resolve active menu id from a pathname.
 * Falls through to the longest-match route in STUDIO_MENU_PRIMARY.
 */
export function resolveActiveMenuId(pathname: string): string | null {
  // Special case: /chat/[id] → 'chat'
  if (pathname.startsWith('/chat')) return 'chat';
  if (pathname.startsWith('/missions/')) return 'assign'; // mission detail nests under Assignments
  if (pathname.startsWith('/activity')) return 'cmd';     // activity panels live under Command Centre
  if (pathname.startsWith('/marketplace')) return 'market';
  if (pathname.startsWith('/admin')) return 'admin';
  if (pathname.startsWith('/playbooks')) return 'assign';
  if (pathname.startsWith('/settings')) return 'settings';
  if (pathname.startsWith('/deliverables')) return 'deliv';

  // Exact-href match
  const direct = STUDIO_MENU_PRIMARY.find((m) => m.href === pathname);
  if (direct) return direct.id;

  // Prefix match
  const prefix = STUDIO_MENU_PRIMARY.find(
    (m) => pathname.startsWith(m.href + '/') || pathname === m.href
  );
  return prefix?.id ?? null;
}
