/**
 * PRD-261 A-12a — SDK area `cards`. Each name draws the host's own card (so its buttons make the
 * host's calls), handed the theme's props as a copy the host may keep: the approval cards checked as
 * the host's chat checks a payload off the wire, with the widget metadata built from where the card
 * came from; the reply, content and chrome cards as they are; CorePage the core page the host hands
 * down, or its children where there is none. The host's cards are stood in for here, recording
 * what they were given; MadeBy runs the host's own line, and AiLabelFence the host's own fence on the
 * host's AI label store (PRD-264 EUAI-11a).
 */
import React from 'react'
import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => {
  const props = new Map<string, Record<string, unknown>>()
  return {
    props,
    record(name: string, value: Record<string, unknown>) {
      props.set(name, value)
    },
  }
})

vi.mock('@/components/widgets/ToolApprovalWidget', () => ({
  ToolApprovalWidget: (props: Record<string, unknown>) => {
    host.record('ToolApprovalWidget', props)
    return <div data-testid="ToolApprovalWidget" />
  },
}))
vi.mock('@/components/widgets/MissionApprovalWidget', () => ({
  MissionApprovalWidget: (props: Record<string, unknown>) => {
    host.record('MissionApprovalWidget', props)
    return <div data-testid="MissionApprovalWidget" />
  },
}))
vi.mock('@/components/chatbot/task-card', () => ({
  TaskCard: (props: Record<string, unknown>) => {
    host.record('TaskCard', props)
    return <div data-testid="TaskCard" />
  },
}))
vi.mock('@/components/chatbot/withdraw-offer-card', () => ({
  WithdrawOfferCard: (props: Record<string, unknown>) => {
    host.record('WithdrawOfferCard', props)
    return <div data-testid="WithdrawOfferCard" />
  },
}))
vi.mock('@/components/chatbot/mission-suggestion-card', () => ({
  MissionSuggestionCard: (props: Record<string, unknown>) => {
    host.record('MissionSuggestionCard', props)
    return <div data-testid="MissionSuggestionCard" />
  },
}))
vi.mock('@/components/chatbot/mission-created-card', () => ({
  MissionCreatedCard: (props: Record<string, unknown>) => {
    host.record('MissionCreatedCard', props)
    return <div data-testid="MissionCreatedCard" />
  },
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: (props: Record<string, unknown>) => {
    host.record('FilePreview', props)
    return <div data-testid="FilePreview" />
  },
}))
vi.mock('@/components/shared/markdown-view', () => ({
  MarkdownView: (props: Record<string, unknown>) => {
    host.record('MarkdownView', props)
    return <div data-testid="MarkdownView" />
  },
}))
vi.mock('@/components/onboarding/trial-exhausted-banner', () => ({
  TrialExhaustedBanner: (props: Record<string, unknown>) => {
    host.record('TrialExhaustedBanner', props)
    return <div data-testid="TrialExhaustedBanner" />
  },
}))
vi.mock('@/components/local/first-run-nudge', () => ({
  FirstRunNudge: () => <div data-testid="FirstRunNudge" />,
}))
vi.mock('@/components/layout/studio-header', () => ({
  StudioHeaderUtilities: () => <div data-testid="StudioHeaderUtilities" />,
}))

import {
  AiLabelFence,
  CorePage,
  FilePreview,
  FirstRunNudge,
  HeaderUtilities,
  MadeBy,
  MarkdownView,
  MissionApproval,
  MissionCreatedCard,
  MissionSuggestionCard,
  TaskCard,
  ToolApproval,
  TrialExhaustedBanner,
  WithdrawOfferCard,
  type MissionApprovalData,
  type ToolApprovalData,
} from '../cards'
import { checkedMissionCard, checkedToolCard } from '../host-cards/approvals'
import { CorePageProvider } from '../host-cards/core-page'
import { MADE_BY_FALLBACK, MADE_BY_SOURCE } from '@/components/workspace/gallery-view/made-by'
import { MISSION_APPROVAL_TITLE, MISSION_SOURCE, TOOL_APPROVAL_TITLE, UNNAMED_ACTION } from '@/lib/chat/approvals'
import { AI_LABEL_FENCE_ATTR } from '@/components/ai-label/ai-label-fence'
import { PAGE_AI_LABEL, WIDGET_AI_LABEL, useAiLabelStore } from '@/lib/ai-label/store'

const GRANT: ToolApprovalData = {
  grant_id: 12101,
  action: 'platform_delete_document',
  message: 'This action (destructive) requires confirmation.',
  permission_level: 'destructive',
  params: { document_id: 'e2e-a12a-doc' },
  risk_tier: 'human_in_the_loop',
  oversight_rationale: 'Deletes a document.',
}

const PLAN: MissionApprovalData = {
  mission_id: 'mission-a12a',
  goal: 'Refund the customer and email them',
  task_count: 2,
  tasks: [
    { title: 'Issue the refund', agent_role: 'ops', sequence: 1 },
    { title: 'Email the customer', agent_role: 'comms', sequence: 2 },
  ],
}

function given(name: string): Record<string, unknown> {
  const props = host.props.get(name)
  if (props === undefined) throw new Error(`${name} was not rendered`)
  return props
}

afterEach(() => {
  host.props.clear()
  useAiLabelStore.setState({ hiddenLabels: [] })
})

describe('PRD-261 A-12a — ToolApproval', () => {
  it("draws the host's widget with the checked payload, its title, its badge and the theme's callbacks", () => {
    const onDecided = vi.fn()
    const onClose = vi.fn()
    render(<ToolApproval data={GRANT} conversationId="chat-1" onDecided={onDecided} onClose={onClose} />)
    const props = given('ToolApprovalWidget')
    expect(props).toMatchObject({ id: 'tool-12101', title: TOOL_APPROVAL_TITLE, onDecided, onClose })
    expect(props.data).toMatchObject(GRANT)
    expect(props.data).not.toBe(GRANT)
    expect(props.metadata).toMatchObject({ source: { type: 'tool', name: GRANT.action }, conversationId: 'chat-1' })
    expect((props.metadata as { createdAt: unknown }).createdAt).toBeInstanceOf(Date)
  })

  it('takes the title, the badge and the time the theme gives; an action-less card is badged as the host badges it', () => {
    const raised = new Date('2026-10-19T10:00:00Z')
    render(<ToolApproval data={GRANT} title="Approve this" source="auto" createdAt={raised} />)
    expect(given('ToolApprovalWidget')).toMatchObject({ title: 'Approve this', metadata: { source: { name: 'auto' }, createdAt: raised } })
    render(<ToolApproval data={{ ...GRANT, action: '' }} />)
    expect(given('ToolApprovalWidget')).toMatchObject({ metadata: { source: { name: UNNAMED_ACTION } } })
  })

  it('refuses a grant id that is not a whole number before any request is made', () => {
    expect(checkedToolCard(GRANT).key).toBe('tool-12101')
    for (const grantId of [1.5, -3, '12/../deny', null]) {
      expect(() => checkedToolCard({ ...GRANT, grant_id: grantId }), String(grantId)).toThrow(TypeError)
    }
  })
})

describe('PRD-261 A-12a — MissionApproval', () => {
  it("draws the host's widget with the checked plan, the Mission badge and the theme's onClose", () => {
    const onClose = vi.fn()
    render(<MissionApproval data={PLAN} onClose={onClose} />)
    const props = given('MissionApprovalWidget')
    expect(props).toMatchObject({ id: 'mission-mission-a12a', title: MISSION_APPROVAL_TITLE, onClose })
    expect(props.metadata).toMatchObject({ source: { type: 'tool', name: MISSION_SOURCE } })
    const data = props.data as MissionApprovalData
    expect(data).toMatchObject({ mission_id: PLAN.mission_id, goal: PLAN.goal, task_count: 2 })
    expect(data.tasks).toEqual(PLAN.tasks.map((task) => expect.objectContaining(task)))
    expect(data.tasks).not.toBe(PLAN.tasks)
  })

  it('refuses a Mission id that could leave its request path', () => {
    expect(checkedMissionCard(PLAN).key).toBe('mission-mission-a12a')
    for (const missionId of ['a/b', '../approve', '', 42]) {
      expect(() => checkedMissionCard({ ...PLAN, mission_id: missionId }), String(missionId)).toThrow(TypeError)
    }
  })
})

describe('PRD-261 A-12a — the chat reply cards', () => {
  it("TaskCard draws the host's ticket card for the theme's card", () => {
    const card = { id: 42, number: '#0042', title: 'Ship it', status: 'in_progress', assigned_agent: 'Atlas' }
    render(<TaskCard card={card} />)
    expect(given('TaskCard')).toEqual({ card })
  })

  it("WithdrawOfferCard hands the host's card a copy of the requests", () => {
    const offer = { message: 'Withdraw these?', requests: [{ grant_id: 7, action: 'delete', reason: 'owner said no' }] }
    render(<WithdrawOfferCard offer={offer} />)
    const handed = given('WithdrawOfferCard').offer as typeof offer
    expect(handed).toEqual(offer)
    expect(handed.requests).not.toBe(offer.requests)
  })

  it("MissionSuggestionCard and MissionCreatedCard draw the host's Mission cards", () => {
    const recentMessages = [{ role: 'user', content: 'Plan the launch' }]
    render(<MissionSuggestionCard goal="Plan the launch" complexity="organ" agentId={3} chatId="chat-1" recentMessages={recentMessages} />)
    const suggestion = given('MissionSuggestionCard')
    expect(suggestion).toEqual({ goal: 'Plan the launch', complexity: 'organ', agentId: 3, chatId: 'chat-1', recentMessages })
    expect(suggestion.recentMessages).not.toBe(recentMessages)
    render(<MissionCreatedCard missionId="mission-a12a" />)
    expect(given('MissionCreatedCard')).toEqual({ missionId: 'mission-a12a' })
  })
})

describe('PRD-261 A-12a — the content cards', () => {
  it("FilePreview draws the host's preview with the theme's props", () => {
    render(<FilePreview url="/api/deliverables/d-1/file" previewType="pdf" filename="plan.pdf" />)
    expect(given('FilePreview')).toEqual({ url: '/api/deliverables/d-1/file', previewType: 'pdf', filename: 'plan.pdf' })
  })

  it("MarkdownView draws the host's renderer with copies of the overrides and the dropped elements", () => {
    const components = { a: () => null }
    const disallowedElements = ['img']
    render(
      <MarkdownView density="compact" skipHtml className="prose" components={components} disallowedElements={disallowedElements}>
        {'# Hello'}
      </MarkdownView>,
    )
    const props = given('MarkdownView')
    expect(props).toMatchObject({ density: 'compact', skipHtml: true, className: 'prose', children: '# Hello' })
    expect(props.components).toEqual(components)
    expect(props.components).not.toBe(components)
    expect(props.disallowedElements).toEqual(disallowedElements)
    expect(props.disallowedElements).not.toBe(disallowedElements)
  })

  it('MadeBy says who made a Deliverable as the host does: the agent, else where it came from, else the fallback', () => {
    const { rerender } = render(<MadeBy agentName="Atlas" sourceType="upload" />)
    expect(screen.getByText('Atlas')).toBeInTheDocument()
    rerender(<MadeBy sourceType="upload" />)
    expect(screen.getByText(MADE_BY_SOURCE.upload)).toBeInTheDocument()
    rerender(<MadeBy />)
    expect(screen.getByText(MADE_BY_FALLBACK)).toBeInTheDocument()
  })
})

describe('PRD-261 A-12a — the chrome cards', () => {
  it("HeaderUtilities draws the Studio header's utilities cluster", () => {
    render(<HeaderUtilities />)
    expect(screen.getByTestId('StudioHeaderUtilities')).toBeInTheDocument()
  })

  it("TrialExhaustedBanner and FirstRunNudge draw the host's own", () => {
    render(<TrialExhaustedBanner errorCode="trial_exhausted" />)
    expect(given('TrialExhaustedBanner')).toEqual({ errorCode: 'trial_exhausted' })
    render(<FirstRunNudge />)
    expect(screen.getByTestId('FirstRunNudge')).toBeInTheDocument()
  })
})

describe('PRD-261 A-12a — CorePage', () => {
  it('renders the core page the host hands down, not its children', () => {
    render(
      <CorePageProvider page={<p>The core page</p>}>
        <CorePage>
          <p>No core page here</p>
        </CorePage>
      </CorePageProvider>,
    )
    expect(screen.getByText('The core page')).toBeInTheDocument()
    expect(screen.queryByText('No core page here')).toBeNull()
  })

  it('renders its children where there is no core page', () => {
    render(
      <CorePage>
        <p>No core page here</p>
      </CorePage>,
    )
    expect(screen.getByText('No core page here')).toBeInTheDocument()
  })
})

describe("PRD-261 W-A5 — AiLabelFence (the host's AI label fence, PRD-264 EUAI-11a)", () => {
  const COMPOSER = 'Message Auto'

  function hide(label: typeof PAGE_AI_LABEL, hidden: boolean) {
    act(() => useAiLabelStore.getState().setHidden(label, hidden))
  }

  it("is the host's fence: a theme's composer pauses while the page's label is hidden, and comes back", () => {
    render(
      <AiLabelFence>
        <textarea aria-label={COMPOSER} />
      </AiLabelFence>,
    )
    const composer = screen.getByRole('textbox', { name: COMPOSER })
    expect(composer.closest(`[${AI_LABEL_FENCE_ATTR}]`)).not.toBeNull()
    expect(composer).toBeEnabled()
    hide(PAGE_AI_LABEL, true)
    expect(composer).toBeDisabled()
    hide(PAGE_AI_LABEL, false)
    expect(composer).toBeEnabled()
  })

  it("hands the host the theme's label and held: the widget's fence follows the widget's label, held stays disabled", () => {
    render(
      <AiLabelFence label={WIDGET_AI_LABEL}>
        <textarea aria-label={COMPOSER} />
      </AiLabelFence>,
    )
    const composer = screen.getByRole('textbox', { name: COMPOSER })
    hide(PAGE_AI_LABEL, true)
    expect(composer).toBeEnabled()
    hide(WIDGET_AI_LABEL, true)
    expect(composer).toBeDisabled()
    render(
      <AiLabelFence held>
        <button type="button">Send</button>
      </AiLabelFence>,
    )
    hide(PAGE_AI_LABEL, false)
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
  })
})
