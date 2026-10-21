/**
 * #1091 — the role select's options used a fixed dark hex, so the popup stayed
 * near-black in light mode. They now read the theme's popover surface.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MemberRow } from '../member-row'
import type { TeamMember } from '../team-types'

const member: TeamMember = {
    id: 7,
    user_id: 'user_7',
    email: 'ada@example.com',
    name: 'Ada',
    role: 'editor',
    joined_at: '2026-01-02T03:04:05Z',
    last_active_at: null,
    tool_runs_30d: 0,
    chats_30d: 0,
    invited_by_email: null,
}

afterEach(cleanup)

describe('MemberRow role select', () => {
    it('paints every option with the popover tokens, never a fixed dark hex', () => {
        render(<MemberRow member={member} onRoleChange={vi.fn()} onRemove={vi.fn()} />)

        const options = screen.getAllByRole('option')
        expect(options).toHaveLength(4)
        for (const option of options) {
            expect(option.className).toContain('bg-popover')
            expect(option.className).toContain('text-popover-foreground')
            expect(option.className).not.toMatch(/bg-\[#/)
            expect(option.className).not.toMatch(/text-(purple|blue|green)-400/)
        }
    })
})
