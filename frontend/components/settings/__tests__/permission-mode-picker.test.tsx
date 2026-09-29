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
