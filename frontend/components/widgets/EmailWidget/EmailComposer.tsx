'use client'

/**
 * EmailComposer Component for PRD-38.2 Extended Widgets
 *
 * Compose and reply to emails with validation.
 * When a replyTo prop is provided, pre-fills To, Subject (Re: prefix), and quotes body.
 */

import { useState, useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import {
  X,
  Send,
  Paperclip,
  ChevronDown,
  ChevronUp,
  Trash2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { EmailFull, EmailDraft } from '../types'

/** Basic email format check — intentionally permissive (user@domain.tld) */
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

interface ValidationErrors {
  to?: string
  subject?: string
}

interface EmailComposerProps {
  replyTo?: EmailFull
  onSend?: (draft: EmailDraft) => Promise<void>
  onDiscard?: () => void
  onMinimize?: () => void
  isMinimized?: boolean
}

export function EmailComposer({
  replyTo,
  onSend,
  onDiscard,
  onMinimize,
  isMinimized = false,
}: EmailComposerProps) {
  // ---------------------------------------------------------------------------
  // Draft state — pre-fill from replyTo when present
  // ---------------------------------------------------------------------------
  const [draft, setDraft] = useState<EmailDraft>(() => {
    if (replyTo) {
      return {
        to: [replyTo.from.email],
        cc: replyTo.cc?.map((addr) => addr.email),
        subject: replyTo.subject.startsWith('Re:')
          ? replyTo.subject
          : `Re: ${replyTo.subject}`,
        body: `\n\n---\nOn ${new Date(replyTo.date).toLocaleString()}, ${replyTo.from.name || replyTo.from.email} wrote:\n\n${replyTo.body}`,
      }
    }
    return {
      to: [],
      subject: '',
      body: '',
    }
  })

  const [showCcBcc, setShowCcBcc] = useState(
    Boolean(replyTo?.cc && replyTo.cc.length > 0)
  )
  const [isSending, setIsSending] = useState(false)
  const [toInput, setToInput] = useState(draft.to.join(', '))
  const [ccInput, setCcInput] = useState(draft.cc?.join(', ') || '')
  const [bccInput, setBccInput] = useState(draft.bcc?.join(', ') || '')
  const [errors, setErrors] = useState<ValidationErrors>({})
  /** True after the first send attempt so we show inline errors immediately */
  const [touched, setTouched] = useState(false)

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  const updateDraft = useCallback(
    (field: keyof EmailDraft, value: unknown) => {
      setDraft((prev) => ({ ...prev, [field]: value }))
    },
    []
  )

  /** Parse comma-separated emails, trimming whitespace and dropping empties */
  const parseEmails = useCallback((input: string): string[] => {
    return input
      .split(',')
      .map((e) => e.trim())
      .filter((e) => e.length > 0)
  }, [])

  /** Returns true when every address in the list passes the basic regex */
  const allEmailsValid = useCallback((addresses: string[]): boolean => {
    return addresses.length > 0 && addresses.every((a) => EMAIL_RE.test(a))
  }, [])

  /** Run validation and return errors (empty object = valid) */
  const validate = useCallback((): ValidationErrors => {
    const next: ValidationErrors = {}
    const toAddresses = parseEmails(toInput)

    if (toAddresses.length === 0) {
      next.to = 'At least one recipient is required'
    } else if (!allEmailsValid(toAddresses)) {
      next.to = 'One or more email addresses are invalid'
    }

    if (!draft.subject.trim()) {
      next.subject = 'Subject is required'
    }

    return next
  }, [toInput, draft.subject, parseEmails, allEmailsValid])

  /** Convenience: can we send right now? */
  const canSend = useCallback((): boolean => {
    const toAddresses = parseEmails(toInput)
    return (
      allEmailsValid(toAddresses) && draft.subject.trim().length > 0
    )
  }, [toInput, draft.subject, parseEmails, allEmailsValid])

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleSend = useCallback(async () => {
    if (!onSend) return
    setTouched(true)

    const validationErrors = validate()
    setErrors(validationErrors)
    if (Object.keys(validationErrors).length > 0) return

    const finalDraft: EmailDraft = {
      to: parseEmails(toInput),
      cc: showCcBcc ? parseEmails(ccInput) : undefined,
      bcc: showCcBcc ? parseEmails(bccInput) : undefined,
      subject: draft.subject.trim(),
      body: draft.body,
    }

    setIsSending(true)
    try {
      await onSend(finalDraft)
    } finally {
      setIsSending(false)
    }
  }, [onSend, toInput, ccInput, bccInput, draft, showCcBcc, validate, parseEmails])

  const handleDiscard = useCallback(() => {
    const hasContent =
      toInput.length > 0 ||
      draft.subject.length > 0 ||
      draft.body.length > 0

    if (hasContent) {
      if (confirm('Discard this draft?')) {
        onDiscard?.()
      }
    } else {
      onDiscard?.()
    }
  }, [toInput, draft, onDiscard])

  // Clear field-level errors as the user types (only after first submit attempt)
  const handleToChange = useCallback(
    (value: string) => {
      setToInput(value)
      if (touched) {
        const addresses = parseEmails(value)
        if (addresses.length > 0 && allEmailsValid(addresses)) {
          setErrors((prev) => ({ ...prev, to: undefined }))
        }
      }
    },
    [touched, parseEmails, allEmailsValid]
  )

  const handleSubjectChange = useCallback(
    (value: string) => {
      updateDraft('subject', value)
      if (touched && value.trim().length > 0) {
        setErrors((prev) => ({ ...prev, subject: undefined }))
      }
    },
    [touched, updateDraft]
  )

  // ---------------------------------------------------------------------------
  // Minimized state
  // ---------------------------------------------------------------------------
  if (isMinimized) {
    return (
      <div
        className={cn(
          'flex items-center justify-between px-3 py-2',
          'bg-muted/50 border-t border-border/30 cursor-pointer'
        )}
        onClick={onMinimize}
      >
        <span className="text-sm font-medium truncate">
          {replyTo ? `Re: ${replyTo.subject}` : 'New Message'}
        </span>
        <ChevronUp className="h-4 w-4 text-muted-foreground" />
      </div>
    )
  }

  // ---------------------------------------------------------------------------
  // Full compose form
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/30">
        <span className="text-sm font-medium">
          {replyTo ? 'Reply' : 'New Message'}
        </span>
        <div className="flex items-center gap-1">
          {onMinimize && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={onMinimize}
              title="Minimize"
            >
              <ChevronDown className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleDiscard}
            title="Discard"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Form */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="px-3 py-2 space-y-3 border-b border-border/30">
          {/* To field */}
          <div>
            <div className="flex items-center gap-2">
              <Label htmlFor="to" className="w-10 text-sm text-muted-foreground">
                To
              </Label>
              <Input
                id="to"
                type="text"
                placeholder="recipient@example.com"
                value={toInput}
                onChange={(e) => handleToChange(e.target.value)}
                className={cn('flex-1 h-8', errors.to && 'border-destructive')}
                aria-invalid={Boolean(errors.to)}
                aria-describedby={errors.to ? 'to-error' : undefined}
              />
              <Button
                variant="ghost"
                size="sm"
                className="h-8 text-xs"
                onClick={() => setShowCcBcc(!showCcBcc)}
              >
                {showCcBcc ? 'Hide' : 'Cc/Bcc'}
              </Button>
            </div>
            {errors.to && (
              <p id="to-error" className="ml-12 mt-1 text-xs text-destructive">
                {errors.to}
              </p>
            )}
          </div>

          {/* Cc/Bcc fields */}
          {showCcBcc && (
            <>
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="cc"
                  className="w-10 text-sm text-muted-foreground"
                >
                  Cc
                </Label>
                <Input
                  id="cc"
                  type="text"
                  placeholder="cc@example.com"
                  value={ccInput}
                  onChange={(e) => setCcInput(e.target.value)}
                  className="flex-1 h-8"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="bcc"
                  className="w-10 text-sm text-muted-foreground"
                >
                  Bcc
                </Label>
                <Input
                  id="bcc"
                  type="text"
                  placeholder="bcc@example.com"
                  value={bccInput}
                  onChange={(e) => setBccInput(e.target.value)}
                  className="flex-1 h-8"
                />
              </div>
            </>
          )}

          {/* Subject field */}
          <div>
            <div className="flex items-center gap-2">
              <Label
                htmlFor="subject"
                className="w-10 text-sm text-muted-foreground"
              >
                Subj
              </Label>
              <Input
                id="subject"
                type="text"
                placeholder="Subject"
                value={draft.subject}
                onChange={(e) => handleSubjectChange(e.target.value)}
                className={cn('flex-1 h-8', errors.subject && 'border-destructive')}
                aria-invalid={Boolean(errors.subject)}
                aria-describedby={errors.subject ? 'subject-error' : undefined}
              />
            </div>
            {errors.subject && (
              <p id="subject-error" className="ml-12 mt-1 text-xs text-destructive">
                {errors.subject}
              </p>
            )}
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 p-3 overflow-hidden">
          <Textarea
            placeholder="Write your message..."
            value={draft.body}
            onChange={(e) => updateDraft('body', e.target.value)}
            className="h-full resize-none"
          />
        </div>

        {/* Footer with actions */}
        <div className="flex items-center justify-between px-3 py-2 border-t border-border/30">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-8"
              title="Attach file (not yet implemented)"
              disabled
            >
              <Paperclip className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-destructive hover:text-destructive"
              onClick={handleDiscard}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Discard
            </Button>
          </div>

          <Button
            size="sm"
            className="h-8"
            onClick={handleSend}
            disabled={isSending || !canSend()}
          >
            <Send className="h-4 w-4 mr-1" />
            {isSending ? 'Sending...' : 'Send'}
          </Button>
        </div>
      </div>
    </div>
  )
}
