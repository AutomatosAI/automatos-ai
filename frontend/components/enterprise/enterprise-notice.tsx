'use client'

/**
 * PRD-277 Wave 5 — a whole-page notice in place of the app (waiting for a workspace, the API out of
 * reach, checking the sign-in), drawn like "Pair this browser" (pair-browser-screen.tsx).
 */
import { Button } from '@/components/ui/button'

interface NoticeAction {
  readonly label: string
  readonly onClick: () => void
}

interface EnterpriseNoticeProps {
  readonly title: string
  readonly body?: string
  readonly action?: NoticeAction
}

export function EnterpriseNotice({ title, body, action }: EnterpriseNoticeProps) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground">
      <section
        aria-labelledby="enterprise-notice-title"
        className="w-full max-w-sm space-y-5 rounded-2xl border border-border bg-card p-6 text-card-foreground"
      >
        <div className="space-y-2">
          <h1 id="enterprise-notice-title" className="text-xl font-semibold">{title}</h1>
          {body && <p className="text-sm text-muted-foreground">{body}</p>}
        </div>
        {action && (
          <Button type="button" variant="outline" className="w-full" onClick={action.onClick}>
            {action.label}
          </Button>
        )}
      </section>
    </main>
  )
}
