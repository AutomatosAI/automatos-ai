'use client'

/**
 * PRD-261 A-04c (§6 "Kill switch and recovery") — /safe-mode, the way out of a theme that keeps
 * failing. Opening it turns safe mode on for this tab (the host loads no theme code here or on
 * any later page of the tab), and the link back carries `?safe_mode=themes` as well, so the
 * next page is safe even where this tab's sessionStorage cannot be written.
 *
 * The page stands alone, outside MainLayout: it draws none of the chrome a theme replaces and
 * calls no backend route, so nothing here can fail the way the theme did. The root layout's boot
 * has already painted the installed theme's classes and prepaint variables on <html> before this
 * page; the constant script first in the page takes them off again and puts Studio's on, before
 * the page paints. A client-side visit, where React never runs that script, does the same in its
 * effect.
 */
import Link from 'next/link'
import { useEffect } from 'react'

import { PageHeader } from '@/components/shared/page-header'
import {
  SAFE_MODE_ADD_CLASSES,
  SAFE_MODE_REMOVE_CLASSES,
  SAFE_MODE_UNPAINT_ID,
  SAFE_MODE_UNPAINT_SCRIPT,
  unpaintThemeBoot,
} from '@/components/theme-host/theme-host-provider'
import { Button } from '@/components/ui/button'
import { requestSafeModeForThisTab, SAFE_MODE_QUERY } from '@/lib/theme-host/safe-mode'

/** Back into the app, still in safe mode. */
const BACK_HREF = `/chat${SAFE_MODE_QUERY}`

const STEPS: readonly string[] = [
  'No installed theme runs in this tab: the app shows Studio in its place.',
  'Safe mode lasts until you close this tab. Other tabs are not changed.',
  'To leave a theme for good, choose Studio or Classic in the style menu at the top of the app.',
]

export default function SafeModePage() {
  useEffect(() => {
    requestSafeModeForThisTab()
    unpaintThemeBoot()
  }, [])

  return (
    <main className="min-h-screen bg-background px-6 py-10">
      <script
        id={SAFE_MODE_UNPAINT_ID}
        data-add={SAFE_MODE_ADD_CLASSES.join(' ')}
        data-remove={SAFE_MODE_REMOVE_CLASSES.join(' ')}
        dangerouslySetInnerHTML={{ __html: SAFE_MODE_UNPAINT_SCRIPT }}
      />
      <div className="mx-auto flex max-w-2xl flex-col gap-6">
        <PageHeader
          eyebrow="Recovery"
          title="Safe"
          titleAccent="mode"
          lede="Themes are off in this tab. Use it to get back in when a theme stops the app from working."
        />
        <ul className="list-disc space-y-2 pl-5 text-sm text-muted-foreground">
          {STEPS.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ul>
        <div>
          <Button asChild>
            <Link href={BACK_HREF} data-testid="safe-mode-back">
              Back to the app
            </Link>
          </Button>
        </div>
      </div>
    </main>
  )
}
