'use client'

/**
 * PRD-277 US-502 — "Your session ended", over the page, when the API answers 401 `session_ended`
 * (the session expired, was ended by an admin, or the person was deactivated). The page stays
 * mounted underneath; the one button keeps the chat box's text (kept-chat-draft.ts) and sends the
 * browser to sign in again, bringing the person back to this page. It cannot be dismissed: every
 * request would be refused until the person signs in again.
 */
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { currentPath, signInStartUrl } from '@/lib/enterprise/auth-api'
import { keepChatDraft } from '@/lib/enterprise/kept-chat-draft'
import { SESSION_ENDED_BODY, SESSION_ENDED_TITLE, SIGN_IN_AGAIN } from '@/lib/enterprise/messages'

interface SessionEndedDialogProps {
  /** Where the browser goes; the address bar by default (a seam for tests). */
  readonly navigate?: (url: string) => void
}

export function SessionEndedDialog({ navigate = (url) => window.location.assign(url) }: SessionEndedDialogProps) {
  const signInAgain = () => {
    keepChatDraft()
    navigate(signInStartUrl(currentPath()))
  }

  return (
    <AlertDialog open>
      <AlertDialogContent className="rounded-2xl">
        <AlertDialogHeader>
          <AlertDialogTitle>{SESSION_ENDED_TITLE}</AlertDialogTitle>
          <AlertDialogDescription>{SESSION_ENDED_BODY}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={signInAgain}>{SIGN_IN_AGAIN}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
