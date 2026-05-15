'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChevronsUpDown, ExternalLink } from 'lucide-react';
import {
  STUDIO_MENU_PRIMARY,
  STUDIO_MENU_FOOTER,
  resolveActiveMenuId,
  type StudioMenuGroup,
} from '@/lib/studio-menu';

/**
 * StudioSidebar — SIDE-B (labelled rail, 232px) per CD's shell delivery.
 *
 * 13 menu items in 3 groups (OPERATIONS / WORKFORCE / WORKSPACE) +
 * workspace switcher + footer (Docs external + Settings) + mini-stats foot.
 *
 * Renders under `.studio` scope only (gated by MainLayout). Wire workspace
 * data + sign-out handlers in via props when integrating with Clerk.
 *
 * PRD shell rollout reference.
 */

const GROUP_ORDER: StudioMenuGroup[] = ['OPERATIONS', 'WORKFORCE', 'WORKSPACE'];

export interface StudioSidebarProps {
  /** Workspace name shown in the switcher pill */
  workspaceName?: string;
  /** Sub-line metadata (e.g. "pilot · 11 op") */
  workspaceMeta?: string;
  /** One-letter mark for the workspace square (falls back to first letter of name) */
  workspaceMark?: string;
  /** Hook for workspace switcher click — leave null to make it non-interactive */
  onWorkspaceClick?: () => void;
  /** Optional alert counts per menu id (shown as right-aligned count). Use the
   *  `alert` key for burnt-orange treatment. e.g. { assign: '!' } or { assign: { count: 3, alert: true } } */
  alerts?: Record<string, string | number>;
  /** Show the mini-stats footer (tick, $/dec, cache). Defaults true. */
  showStats?: boolean;
}

export function StudioSidebar({
  workspaceName = 'Automatos AI',
  workspaceMeta = 'pilot · 11 op',
  workspaceMark,
  onWorkspaceClick,
  alerts = {},
  showStats = true,
}: StudioSidebarProps) {
  const pathname = usePathname();
  const activeId = pathname ? resolveActiveMenuId(pathname) : null;
  const mark = workspaceMark ?? workspaceName.slice(0, 1).toUpperCase();

  return (
    <aside className="sh-side" data-tour="sidebar-studio">
      {/* Brand row */}
      <div className="sh-brand-row">
        <Link href="/" className="sh-brand" style={{ fontSize: 17 }}>
          <span className="sh-glyph" />
          <span>automatos</span>
        </Link>
        <span
          style={{
            marginLeft: 'auto',
            fontFamily: 'var(--font-geist-mono, monospace)',
            fontSize: 9,
            color: 'hsl(var(--muted-foreground))',
            letterSpacing: '0.08em',
          }}
        >
          v0.11
        </span>
      </div>

      {/* Workspace switcher pill */}
      <button
        type="button"
        className="sh-ws-card"
        onClick={onWorkspaceClick}
        aria-label="Switch workspace"
      >
        <span className="sh-mk">{mark}</span>
        <div style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
          <div className="sh-nm">{workspaceName}</div>
          <div className="sh-meta">{workspaceMeta}</div>
        </div>
        <ChevronsUpDown
          style={{ width: 13, height: 13, color: 'hsl(var(--muted-foreground))', strokeWidth: 1.6 }}
        />
      </button>

      {/* Primary menu — grouped */}
      <nav style={{ flex: 1, overflow: 'hidden auto' }} aria-label="Main">
        {GROUP_ORDER.map((group) => (
          <div key={group}>
            <div className="sh-group">{group}</div>
            {STUDIO_MENU_PRIMARY.filter((m) => m.group === group).map((m) => {
              const Icon = m.icon;
              const isActive = activeId === m.id;
              const alert = alerts[m.id];
              return (
                <Link
                  key={m.id}
                  href={m.href}
                  className={`sh-item${isActive ? ' active' : ''}`}
                  aria-current={isActive ? 'page' : undefined}
                  title={m.desc}
                >
                  <Icon className="sh-ic" strokeWidth={1.6} />
                  <span>{m.label}</span>
                  {alert != null && (
                    <span
                      className={`sh-ct${typeof alert === 'string' && alert === '!' ? ' alert' : ''}`}
                    >
                      {alert}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer — Docs external + Settings */}
      <div className="sh-footer">
        {STUDIO_MENU_FOOTER.map((m) => {
          const Icon = m.icon;
          const isActive = activeId === m.id;
          if (m.external) {
            return (
              <a
                key={m.id}
                href={m.href}
                target="_blank"
                rel="noopener noreferrer"
                className="sh-item"
                title={m.desc}
              >
                <Icon className="sh-ic" strokeWidth={1.6} />
                <span>{m.label}</span>
                <ExternalLink
                  style={{
                    width: 11,
                    height: 11,
                    color: 'hsl(var(--muted-foreground))',
                    marginLeft: 'auto',
                  }}
                />
              </a>
            );
          }
          return (
            <Link
              key={m.id}
              href={m.href}
              className={`sh-item${isActive ? ' active' : ''}`}
              aria-current={isActive ? 'page' : undefined}
              title={m.desc}
            >
              <Icon className="sh-ic" strokeWidth={1.6} />
              <span>{m.label}</span>
            </Link>
          );
        })}

        {showStats && (
          <div className="sh-mini-stats">
            <div className="sh-row">
              <span>tick</span>
              <span>5s</span>
            </div>
            <div className="sh-row">
              <span>$/dec</span>
              <span className="sh-ok">$0.0027</span>
            </div>
            <div className="sh-row">
              <span>cache</span>
              <span className="sh-ok">68%</span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
