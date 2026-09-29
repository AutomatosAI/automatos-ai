/**
 * Session permission modes — the four Claude Code users already know, on
 * Settings → Session mode (the workspace default) and on an agent (its override).
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import {
  PERMISSION_MODES,
  PermissionModePicker,
  isPermissionMode,
  permissionModeLabel,
} from '@/components/settings/PermissionModePicker'

describe('PermissionModePicker', () => {
  it('offers the four modes in Claude Code order and names them as Claude Code does', () => {
    expect(PERMISSION_MODES.map((m) => m.label)).toEqual(['Manual', 'Edit automatically', 'Plan', 'Auto'])
    expect(permissionModeLabel('auto')).toBe('Auto')
    expect(isPermissionMode('bypassPermissions')).toBe(false)
  })

  it('shows the workspace default and saves the one picked', () => {
    const onChange = vi.fn()
    render(<PermissionModePicker value="auto" onChange={onChange} />)
    const radio = (id: string) => screen.getByTestId(`permission-mode-${id}`).querySelector('input') as HTMLInputElement
    expect(radio('auto').checked).toBe(true)
    expect(radio('manual').checked).toBe(false)
    fireEvent.click(radio('plan'))
    expect(onChange).toHaveBeenCalledWith('plan')
  })
})

describe('Plan on a CLI without a plan mode', () => {
  it('names exactly the CLIs whose host preset has a plan mode', async () => {
    const fs = await import('node:fs')
    const path = await import('node:path')
    const presets = fs.readFileSync(
      path.resolve(__dirname, '../../../../services/cli-host/automatos_cli_host/presets.py'),
      'utf8',
    )
    const withPlan = presets
      .split(/\n(?=[A-Z_]+ = CliPreset\()/)
      .filter((block) => /^[A-Z_]+ = CliPreset\(/.test(block) && /\n\s+plan_stance=\(/.test(block))
      .map((block) => /\bid="([^"]+)"/.exec(block)?.[1])
    const { PLAN_MODE_CLIS } = await import('@/components/settings/PermissionModePicker')
    expect(withPlan).toEqual(PLAN_MODE_CLIS)
  })

  it('tells a Codex agent that Plan runs as Edit automatically, and says nothing for Claude Code', async () => {
    const { PermissionModeSelect } = await import('@/components/settings/PermissionModePicker')
    const { unmount } = render(<PermissionModeSelect value="plan" provider="codex" onChange={vi.fn()} />)
    expect(screen.getByTestId('plan-mode-fallback').textContent).toContain('Edit automatically')
    unmount()
    render(<PermissionModeSelect value="plan" provider="claude" onChange={vi.fn()} />)
    expect(screen.queryByTestId('plan-mode-fallback')).toBeNull()
  })
})
