'use client'

/**
 * PRD-268 US-C20c (§6.11.4, K10) — the approval view's parts on the host's two approval cards
 * (ToolApprovalWidget, MissionApprovalWidget): who asked, the action, its target, the exact
 * preview and the rule that asked. They render inside the card, so they sit inside its protected
 * surface (components/theme-host/protected-surface.tsx) and every theme gets them at once.
 *
 * The rule is read before the buttons: the parts come before them in the DOM, and the card's body
 * is a group whose aria-describedby names the rule. The preview is the call's own payload,
 * rendered as text (never as HTML or markdown), whatever its kind. No part carries a data attribute
 * of its own, so no theme selector can aim at one by name inside the protected surface.
 *
 * Keys (K10): while the card's body, or a control inside it, has focus, 1 refuses and 3 allows
 * once, each only when the view offers that answer; 2 does nothing (Always allow is not built,
 * §12-Q17). The handler sits on the body, never on the document, and ignores a key typed into a
 * field (the Mission card's agent box); the buttons announce their key (aria-keyshortcuts) and the
 * body shows a ring while it holds focus. A card without a view is exactly today's: its body takes
 * no role, focus, label or keys.
 */

import { useId, useMemo, type HTMLAttributes, type KeyboardEvent } from 'react'
import { Scale } from 'lucide-react'
import { approvalViewOf } from '@/lib/chat/approvals'
import type { ApprovalAnswer, ApprovalView } from './types'

/** The key that refuses (Deny, Reject) and the key that allows once (Approve & run). */
export const REFUSE_KEY = '1'
export const ALLOW_ONCE_KEY = '3'
/** Who asked, before the name. */
export const ASKED_BY_LABEL = 'Asked by'
/** Both cards' body, today's class: a card without a view keeps exactly this. */
export const APPROVAL_BODY_CLASS = 'flex flex-col gap-3 p-3'

/** The ring shows whenever the keys would answer: the body or a control inside it has focus, by mouse too. */
const FOCUS_CLASS = 'rounded-md outline-none focus-within:ring-2 focus-within:ring-ring'
const TYPING_TAGS: ReadonlySet<string> = new Set(['INPUT', 'TEXTAREA', 'SELECT'])
const TYPING_ROLES: ReadonlySet<string> = new Set(['textbox', 'searchbox', 'combobox', 'spinbutton'])

/** What the card's keys do: the same calls its buttons make, and when they may not. */
export interface ApprovalKeyActions {
  readonly refuse: () => unknown
  readonly allowOnce: () => unknown
  /** A call is in flight or the card is answered: the keys do nothing, as the buttons can't be clicked. */
  readonly disabled: boolean
}

type ButtonKey = Pick<HTMLAttributes<HTMLButtonElement>, 'aria-keyshortcuts'>

/** A key typed into a field is the field's, never an answer. */
function isTypingInto(target: EventTarget): boolean {
  if (!(target instanceof HTMLElement)) return false
  return target.isContentEditable || TYPING_TAGS.has(target.tagName) || TYPING_ROLES.has(target.getAttribute('role') ?? '')
}

function isPlainKey(event: KeyboardEvent<HTMLElement>): boolean {
  return !event.repeat && !event.altKey && !event.ctrlKey && !event.metaKey && !event.nativeEvent.isComposing
}

/** The call a key makes when the view offers its answer: 1 refuses, 3 allows once, any other key (2 included) none. */
function answerFor(key: string, actions: ApprovalKeyActions, answers: readonly ApprovalAnswer[]): (() => unknown) | null {
  if (key === REFUSE_KEY && answers.includes('refuse')) return actions.refuse
  if (key === ALLOW_ONCE_KEY && answers.includes('allow_once')) return actions.allowOnce
  return null
}

/** The body's keydown: 1 refuses, 3 allows once, each only when `answers` offers it; every other key passes through. */
export function approvalKeyHandler(actions: ApprovalKeyActions, answers: readonly ApprovalAnswer[]) {
  return (event: KeyboardEvent<HTMLElement>): void => {
    if (actions.disabled || !isPlainKey(event) || isTypingInto(event.target)) return
    const answer = answerFor(event.key, actions, answers)
    if (answer === null) return
    event.preventDefault()
    event.stopPropagation()
    void answer()
  }
}

export interface ApprovalCardBody {
  /** The view, checked (lib/chat/approvals.ts): the card may come straight off the wire. */
  readonly view: ApprovalView | undefined
  /** The rule's element id, which the body's aria-describedby names. */
  readonly ruleId: string
  /** The body's props: today's class alone without a view. */
  readonly body: HTMLAttributes<HTMLDivElement>
  /** The refusing button's and the allowing button's key, announced; nothing without a view or its answer. */
  readonly refuseKey: ButtonKey
  readonly allowOnceKey: ButtonKey
}

function buttonKey(view: ApprovalView | undefined, answer: ApprovalAnswer, key: string): ButtonKey {
  return view?.answers.includes(answer) ? { 'aria-keyshortcuts': key } : {}
}

/**
 * The card's body: with a view that checks out for `cardId`, a focusable group named by `label` and
 * the action, described by the rule and answering keys 1 and 3; without one, today's class alone.
 */
export function useApprovalCardBody(rawView: unknown, cardId: string, label: string, actions: ApprovalKeyActions): ApprovalCardBody {
  const ruleId = useId()
  const view = useMemo(() => approvalViewOf(rawView, cardId), [rawView, cardId])
  const keys = { refuseKey: buttonKey(view, 'refuse', REFUSE_KEY), allowOnceKey: buttonKey(view, 'allow_once', ALLOW_ONCE_KEY) }
  if (view === undefined) return { view, ruleId, body: { className: APPROVAL_BODY_CLASS }, ...keys }
  const body: HTMLAttributes<HTMLDivElement> = {
    className: `${APPROVAL_BODY_CLASS} ${FOCUS_CLASS}`,
    role: 'group',
    tabIndex: 0,
    'aria-label': `${label}: ${view.action.summary}`,
    'aria-describedby': ruleId,
    onKeyDown: approvalKeyHandler(actions, view.answers),
  }
  return { view, ruleId, body, ...keys }
}

function TargetLine({ target }: { target: NonNullable<ApprovalView['target']> }) {
  return (
    <p className="break-words text-muted-foreground">
      → {target.label}
      {target.detail ? ` · ${target.detail}` : ''}
    </p>
  )
}

/** Who asked, the action, its target, the exact preview and the rule, in that order. */
export function ApprovalViewParts({ view, ruleId }: { view: ApprovalView; ruleId: string }) {
  return (
    <div className="space-y-1.5 rounded border border-border px-2 py-1.5 text-xs">
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {ASKED_BY_LABEL} {view.asked_by.name} · {view.action.verb}
      </p>
      <p className="break-words">{view.action.summary}</p>
      {view.target && <TargetLine target={view.target} />}
      {view.preview && (
        <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded bg-muted/40 px-2 py-1 font-mono text-xs">
          {view.preview.body}
        </pre>
      )}
      <p id={ruleId} className="flex items-start gap-1.5 font-medium">
        <Scale aria-hidden="true" className="mt-px h-3.5 w-3.5 shrink-0" />
        {view.rule.sentence}
      </p>
    </div>
  )
}
