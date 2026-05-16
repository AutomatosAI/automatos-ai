'use client';

import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import {
  STUDIO_MENU_PRIMARY,
  STUDIO_PAGE_TABS,
  resolveActiveMenuId,
} from '@/lib/studio-menu';

/**
 * StudioPageTabs — sub-nav strip beneath the StudioHeader. Reads the active
 * menu id from the pathname and looks up STUDIO_PAGE_TABS for the [name,
 * count, hrefOverride?] tuples. Active tab is driven by `?tab=` (defaults
 * to the first tab) OR by the current pathname matching a tab's
 * hrefOverride (so e.g. Deliverables → Explorer reads active on
 * `/deliverables/explorer`).
 *
 * Renders nothing when there are no tabs registered for the active route.
 */

function slugify(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

export function StudioPageTabs() {
  const pathname = usePathname() ?? '';
  const searchParams = useSearchParams();
  const menuId = resolveActiveMenuId(pathname);
  if (!menuId) return null;

  const tabs = STUDIO_PAGE_TABS[menuId];
  if (!tabs || tabs.length === 0) return null;

  // Base href for `?tab=` tabs — always the menu's canonical href so a
  // sub-route (e.g. /deliverables/explorer) doesn't leave you stuck
  // appending ?tab=… to its own URL.
  const menuBase = STUDIO_MENU_PRIMARY.find((m) => m.id === menuId)?.href ?? pathname;

  // Active tab resolution: prefer a tab whose hrefOverride matches the
  // current pathname; otherwise fall through to `?tab=<slug>` (default
  // first tab if absent).
  const overrideActive = tabs.find(([, , href]) => href && pathname.startsWith(href));
  const activeTab = overrideActive
    ? slugify(overrideActive[0])
    : (searchParams?.get('tab') ?? slugify(tabs[0][0]));

  return (
    <nav className="sh-tabs" aria-label="Page sections">
      {tabs.map(([label, count, hrefOverride]) => {
        const slug = slugify(label);
        const isActive = slug === activeTab;
        const href = hrefOverride ?? `${menuBase}?tab=${slug}`;
        return (
          <Link
            key={slug}
            href={href as any}
            className={`sh-tab${isActive ? ' active' : ''}`}
            aria-current={isActive ? 'page' : undefined}
          >
            <span>{label}</span>
            {count > 0 && <span className="sh-tab-ct">{count}</span>}
          </Link>
        );
      })}
    </nav>
  );
}
