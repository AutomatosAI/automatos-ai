/**
 * PRD-238 S3 — the activity trail: every call stays visible with its state,
 * summary and duration; skipped calls close their line; caps are said aloud.
 */
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ActivityTrail, LimitReachedNote } from '../activity-trail'
import type { ToolCall } from '@/types'

const label = (tc: ToolCall) => tc.toolName.replace(/_/g, ' ')

afterEach(cleanup)

describe('ActivityTrail', () => {
  it('renders one line per call, in order, with state, summary and duration', () => {
    const calls: ToolCall[] = [
      { toolCallId: '1', toolName: 'platform_get_activity_feed', state: 'completed', summary: '12 items', durationMs: 200, input: { limit: 12 } },
      { toolCallId: '2', toolName: 'platform_get_agent', state: 'completed', skipped: true, summary: 'Skipped — already ran with the same input' },
      { toolCallId: '3', toolName: 'platform_fleet_status', state: 'error', error: 'denied' },
      { toolCallId: '4', toolName: 'platform_get_task', state: 'running' },
    ]
    render(<ActivityTrail toolCalls={calls} formatLabel={label} />)
    const items = screen.getAllByRole('listitem')
    expect(items).toHaveLength(4)
    expect(items[0]).toHaveTextContent('platform get activity feed')
    expect(items[0]).toHaveTextContent('12 items')
    expect(items[0]).toHaveTextContent('0.2 s')
    expect(screen.getByLabelText('Skipped')).toBeInTheDocument()
    expect(items[2]).toHaveTextContent('platform fleet status failed')
    expect(items[2]).toHaveTextContent('denied')
    expect(screen.getByLabelText('Running')).toBeInTheDocument()
    // the input is available on expand
    expect(items[0].querySelector('pre')).toHaveTextContent('"limit": 12')
    expect(items[3].querySelector('pre')).toBeNull()
  })

  it('renders nothing for an empty trail', () => {
    const { container } = render(<ActivityTrail toolCalls={[]} formatLabel={label} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('lists progress lines from a long-running tool after the calls', () => {
    render(<ActivityTrail toolCalls={[]} formatLabel={label} progress={['Bob is working on #92 · 10 s', 'Bob is working on #92 · 20 s · last tool: Bash']} />)
    const lines = screen.getAllByTestId('progress-line')
    expect(lines).toHaveLength(2)
    expect(lines[1]).toHaveTextContent('last tool: Bash')
  })

  it('says when a cap ended the turn', () => {
    render(<LimitReachedNote limit={{ limit: 'max_tool_iterations', value: 10, message: 'I reached the maximum of 10 tool steps.' }} />)
    expect(screen.getByRole('status')).toHaveTextContent('maximum of 10 tool steps')
  })
})
