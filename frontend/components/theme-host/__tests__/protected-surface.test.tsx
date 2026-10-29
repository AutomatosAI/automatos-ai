/**
 * PRD-261 A-12c — protected surfaces. The host and the inner wrapper both carry every guard
 * declaration inline with `!important`, and the two approval widgets render their card inside
 * one, wherever they are drawn (the canvas, a chat reply), with the card's buttons inside the
 * guard. The cascade itself (a hostile theme sheet losing to the guard) is a browser fact:
 * e2e/browser/specs/protected-surfaces.spec.ts proves it in Chromium and WebKit. PRD-268 US-C20c:
 * the approval view's rule and preview render inside the guard too, with no hook of their own.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hooks/use-approval-grants', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: vi.fn() }),
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: vi.fn() }),
}))

import { MissionApprovalWidget } from '@/components/widgets/MissionApprovalWidget'
import { ToolApprovalWidget } from '@/components/widgets/ToolApprovalWidget'
import type { ApprovalView, WidgetMetadata } from '@/components/widgets/types'

import {
  guardProtectedElement,
  PROTECTED_SURFACE_ATTR,
  PROTECTED_SURFACE_GUARD,
  PROTECTED_SURFACE_INNER_ATTR,
  ProtectedSurface,
  type ProtectedSurfaceName,
} from '../protected-surface'

const IMPORTANT = 'important'
/** PA-15's five: the ones a theme would use to hide a card or switch off its buttons. */
const PA15_PROPERTIES = ['display', 'visibility', 'opacity', 'pointer-events', 'position']
const APPROVE_BUTTON = /Approve & run/
const metadata: WidgetMetadata = { source: { type: 'tool', name: 'platform_action' }, createdAt: new Date() }
/** PRD-268 US-C20c: the approval view's rule and preview, the parts a theme must never hide. */
const DELETE_RULE = "Ask before anything that deletes or can't be undone"
const PLAN_RULE = 'Ask before any Mission or task runs'
const TOOL_PREVIEW = 'document_id: 12'
const PLAN_PREVIEW = '1. Issue refund'
function viewOf(grantId: string, verb: string, sentence: string, body: string): ApprovalView {
  return {
    grant_id: grantId,
    asked_by: { kind: 'auto', name: 'Auto' },
    action: { verb, summary: 'What the owner approves' },
    preview: { kind: 'text', body },
    rule: { id: 'rule', sentence },
    answers: ['refuse', 'allow_once'],
  }
}

/**
 * The guarded properties the CI jsdom's CSSStyleDeclaration does not know, so setProperty
 * drops them (zoom included, seen in CI 19 Oct): the browser lane checks them
 * (e2e/browser/specs/protected-surfaces.spec.ts).
 */
const JSDOM_UNKNOWN = new Set(['-webkit-mask', 'zoom'])
/** The ones a theme moves, shrinks, masks or skips a card with, beyond PA-15's five (review M2). */
const MOVE_AND_HIDE_PROPERTIES = ['translate', 'scale', 'rotate', 'offset-path', 'mask', '-webkit-mask', 'content-visibility', 'zoom']

/** A part holds no data attribute of its own: nothing a theme selector could name it by. */
function expectNoHookOfItsOwn(element: HTMLElement): void {
  expect(element.getAttributeNames().filter((name) => name.startsWith('data-'))).toEqual([])
}

function expectGuarded(element: HTMLElement): void {
  const checked = Object.entries(PROTECTED_SURFACE_GUARD).filter(([property]) => !JSDOM_UNKNOWN.has(property))
  for (const [property, value] of checked) {
    expect({ property, value: element.style.getPropertyValue(property) }).toEqual({ property, value })
    expect({ property, priority: element.style.getPropertyPriority(property) }).toEqual({ property, priority: IMPORTANT })
  }
}

/** The host and inner wrapper around `inside`, both guarded, the host named `surface`. */
function expectInsideSurface(inside: HTMLElement, surface: ProtectedSurfaceName): void {
  const inner = inside.closest<HTMLElement>(`[${PROTECTED_SURFACE_INNER_ATTR}]`)
  const host = inside.closest<HTMLElement>(`[${PROTECTED_SURFACE_ATTR}]`)
  expect(inner).not.toBeNull()
  expect(host?.getAttribute(PROTECTED_SURFACE_ATTR)).toBe(surface)
  expect(inner?.parentElement).toBe(host)
  expectGuarded(host as HTMLElement)
  expectGuarded(inner as HTMLElement)
}

describe('ProtectedSurface (PRD-261 A-12c)', () => {
  it('guards PA-15\'s five properties at least', () => {
    expect(Object.keys(PROTECTED_SURFACE_GUARD)).toEqual(expect.arrayContaining(PA15_PROPERTIES))
    expect(PROTECTED_SURFACE_GUARD.display).not.toBe('none')
    expect(PROTECTED_SURFACE_GUARD.visibility).toBe('visible')
    expect(PROTECTED_SURFACE_GUARD.opacity).toBe('1')
    expect(PROTECTED_SURFACE_GUARD['pointer-events']).toBe('auto')
    // static voids every top/left a theme sets: no off-screen move
    expect(PROTECTED_SURFACE_GUARD.position).toBe('static')
  })

  it('guards every property that moves, shrinks, masks or skips the card on its own', () => {
    expect(Object.keys(PROTECTED_SURFACE_GUARD)).toEqual(expect.arrayContaining(MOVE_AND_HIDE_PROPERTIES))
    expect(PROTECTED_SURFACE_GUARD['content-visibility']).toBe('visible')
    expect(PROTECTED_SURFACE_GUARD.zoom).toBe('1')
    expect(PROTECTED_SURFACE_GUARD.mask).toBe('none')
    expect(PROTECTED_SURFACE_GUARD['-webkit-mask']).toBe('none')
  })

  it('puts the guard inline, important, on the host and on the inner wrapper', () => {
    render(
      <ProtectedSurface surface="tool-approval">
        <button type="button">inside</button>
      </ProtectedSurface>,
    )
    expectInsideSurface(screen.getByRole('button', { name: 'inside' }), 'tool-approval')
  })

  it('keeps the guard across a re-render (React never rewrites the style attribute)', () => {
    const view = render(
      <ProtectedSurface surface="mission-approval">
        <span>first</span>
      </ProtectedSurface>,
    )
    view.rerender(
      <ProtectedSurface surface="mission-approval">
        <span>second</span>
      </ProtectedSurface>,
    )
    expectInsideSurface(screen.getByText('second'), 'mission-approval')
  })

  it('does nothing when the ref callback is called with null (unmount)', () => {
    expect(() => guardProtectedElement(null)).not.toThrow()
  })

  it('renders the tool approval card, its buttons included, inside a guarded surface', () => {
    render(
      <ToolApprovalWidget
        id="w-tool"
        title="Action approval needed"
        data={{ grant_id: 7, action: 'platform_delete_document' }}
        metadata={metadata}
      />,
    )
    expectInsideSurface(screen.getByRole('button', { name: APPROVE_BUTTON }), 'tool-approval')
    expectInsideSurface(screen.getByRole('button', { name: /Deny/ }), 'tool-approval')
  })

  it('renders the Mission approval card, its buttons included, inside a guarded surface', () => {
    render(
      <MissionApprovalWidget
        id="w-mission"
        title="Mission plan — approval needed"
        data={{ mission_id: 'm-1', goal: 'Refund the customer', task_count: 1, tasks: [{ title: 'Issue refund' }] }}
        metadata={metadata}
      />,
    )
    expectInsideSurface(screen.getByRole('button', { name: APPROVE_BUTTON }), 'mission-approval')
    expectInsideSurface(screen.getByRole('button', { name: /Reject/ }), 'mission-approval')
  })

  it("renders the tool card's approval view, its rule and its preview, inside the guarded surface (PRD-268 US-C20c)", () => {
    render(
      <ToolApprovalWidget
        id="w-tool"
        title="Action approval needed"
        data={{ grant_id: 7, action: 'platform_delete_document', approval_view: viewOf('7', 'delete', DELETE_RULE, TOOL_PREVIEW) }}
        metadata={metadata}
      />,
    )
    for (const part of [screen.getByText(DELETE_RULE), screen.getByText(TOOL_PREVIEW)]) {
      expectInsideSurface(part, 'tool-approval')
      expectNoHookOfItsOwn(part)
    }
    expectInsideSurface(screen.getByRole('group', { name: /^Action approval needed/ }), 'tool-approval')
  })

  it("renders the Mission card's approval view, its rule and its preview, inside the guarded surface (PRD-268 US-C20c)", () => {
    render(
      <MissionApprovalWidget
        id="w-mission"
        title="Mission plan — approval needed"
        data={{
          mission_id: 'm-1',
          goal: 'Refund the customer',
          task_count: 1,
          tasks: [{ title: 'Issue refund' }],
          approval_view: viewOf('m-1', 'run', PLAN_RULE, PLAN_PREVIEW),
        }}
        metadata={metadata}
      />,
    )
    for (const part of [screen.getByText(PLAN_RULE), screen.getByText(PLAN_PREVIEW)]) {
      expectInsideSurface(part, 'mission-approval')
      expectNoHookOfItsOwn(part)
    }
    expectInsideSurface(screen.getByRole('group', { name: /^Mission plan — approval needed/ }), 'mission-approval')
  })
})
