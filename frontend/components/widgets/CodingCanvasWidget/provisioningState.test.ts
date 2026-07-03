/**
 * PRD-170 S2 — canvas provisioning state machine (vitest).
 *
 * The honest canvas-open lifecycle: idle → provisioning → ready (cold vs warm)
 * or failed-with-retry. Drives the provisioning banner + retry affordance.
 */
import { describe, it, expect } from 'vitest'

import {
  initialProvisioningState,
  beginProvisioning,
  resolveProvisioning,
  canRetry,
  provisioningLabel,
} from './provisioningState'

describe('provisioning state machine', () => {
  it('starts idle', () => {
    expect(initialProvisioningState.phase).toBe('idle')
    expect(initialProvisioningState.attempts).toBe(0)
  })

  it('beginProvisioning enters provisioning and counts the attempt', () => {
    const s = beginProvisioning(initialProvisioningState)
    expect(s.phase).toBe('provisioning')
    expect(s.attempts).toBe(1)
    expect(s.error).toBeNull()
  })

  it('a successful cold-provision open → ready + coldProvisioned', () => {
    let s = beginProvisioning(initialProvisioningState)
    s = resolveProvisioning(s, { success: true, provisioned: true })
    expect(s.phase).toBe('ready')
    expect(s.coldProvisioned).toBe(true)
    expect(provisioningLabel(s)).toBe('Workspace provisioned')
  })

  it('a successful warm open → ready, not cold-provisioned', () => {
    let s = beginProvisioning(initialProvisioningState)
    s = resolveProvisioning(s, { success: true, provisioned: false })
    expect(s.phase).toBe('ready')
    expect(s.coldProvisioned).toBe(false)
    expect(provisioningLabel(s)).toBe('Session ready')
  })

  it('a failed open → failed with an honest error + retry offered', () => {
    let s = beginProvisioning(initialProvisioningState)
    s = resolveProvisioning(s, { success: false, error: 'worker unreachable' })
    expect(s.phase).toBe('failed')
    expect(s.error).toBe('worker unreachable')
    expect(canRetry(s)).toBe(true)
    expect(provisioningLabel(s)).toBe('worker unreachable')
  })

  it('failure without a message still surfaces a default error', () => {
    let s = beginProvisioning(initialProvisioningState)
    s = resolveProvisioning(s, { success: false })
    expect(s.error).toBeTruthy()
    expect(s.phase).toBe('failed')
  })

  it('retry re-enters provisioning and increments attempts; caps at maxAttempts', () => {
    let s = beginProvisioning(initialProvisioningState) // attempt 1
    s = resolveProvisioning(s, { success: false, error: 'x' })
    expect(canRetry(s, 3)).toBe(true)

    s = beginProvisioning(s) // attempt 2
    s = resolveProvisioning(s, { success: false, error: 'x' })
    s = beginProvisioning(s) // attempt 3
    s = resolveProvisioning(s, { success: false, error: 'x' })
    expect(s.attempts).toBe(3)
    expect(canRetry(s, 3)).toBe(false) // no more retries past the cap
  })

  it('retry label reflects the attempt number', () => {
    let s = beginProvisioning(initialProvisioningState)
    s = resolveProvisioning(s, { success: false, error: 'x' })
    s = beginProvisioning(s)
    expect(provisioningLabel(s)).toContain('Retrying')
    expect(provisioningLabel(s)).toContain('2')
  })

  it('is immutable — the initial state is never mutated', () => {
    const before = JSON.stringify(initialProvisioningState)
    beginProvisioning(initialProvisioningState)
    resolveProvisioning(initialProvisioningState, { success: true })
    expect(JSON.stringify(initialProvisioningState)).toBe(before)
  })
})
