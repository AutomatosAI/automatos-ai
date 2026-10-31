/**
 * PRD-268 §6.9 — the settings answers checked off the wire (the effective mode among them), the workspace switch as
 * the workspace holds it, and each cap's range as GET/PUT /api/proactive/settings accept it: none looser than the
 * platform's, the shortest gap between check-ins at least the platform's and no longer than the longest.
 */
import { describe, expect, it } from 'vitest'
import {
  CAP_NAMES,
  DEFAULT_CAPS_CHANGE,
  canManageProactive,
  capBounds,
  capFits,
  capValues,
  changedCaps,
  effectiveModeOf,
  proactiveMeOf,
  proactiveSettingsOf,
  wholeNumber,
  withChange,
  workspaceSwitchOf,
  type ProactiveSettings,
} from '../settings'

const PLATFORM = {
  max_model_runs_per_day: 48,
  max_suggestions_per_run: 3,
  max_new_per_person_per_day: 10,
  max_open_per_person: 20,
  max_open_per_agent: 3,
  min_interval_minutes: 30,
  max_interval_minutes: 1440,
}
const SETTINGS: ProactiveSettings = {
  enabled: false,
  caps: { ...PLATFORM, max_open_per_agent: 1 },
  platform_caps: PLATFORM,
  effective_mode: 'off',
}

describe('the answers', () => {
  it('reads the settings answer, and refuses one it cannot read whole', () => {
    expect(proactiveSettingsOf({ enabled: true, caps: PLATFORM, platform_caps: PLATFORM, effective_mode: 'shadow', extra: 1 })).toEqual({
      enabled: true,
      caps: PLATFORM,
      platform_caps: PLATFORM,
      effective_mode: 'shadow',
    })
    // A PUT's answer carries no mode: unknown, not off.
    expect(proactiveSettingsOf({ enabled: true, caps: PLATFORM, platform_caps: PLATFORM }).effective_mode).toBeNull()
    expect(() => proactiveSettingsOf({ enabled: 'yes', caps: PLATFORM, platform_caps: PLATFORM })).toThrow()
    expect(() => proactiveSettingsOf({ enabled: true, caps: { ...PLATFORM, max_open_per_agent: 1.5 }, platform_caps: PLATFORM })).toThrow()
    expect(() => proactiveSettingsOf({ enabled: true, caps: PLATFORM })).toThrow()
    expect(() => proactiveSettingsOf(null)).toThrow()
  })

  it("reads the person's answer", () => {
    expect(proactiveMeOf({ paused: true, quiet_start: '22:00', quiet_end: '07:00', timezone: null, effective_mode: 'live' })).toEqual({
      paused: true,
      quiet_start: '22:00',
      quiet_end: '07:00',
      timezone: null,
      effective_mode: 'live',
    })
    expect(() => proactiveMeOf({ paused: 'no' })).toThrow()
  })

  it('reads only the three modes; anything else is unknown', () => {
    expect(['off', 'shadow', 'live'].map(effectiveModeOf)).toEqual(['off', 'shadow', 'live'])
    expect([undefined, null, 'sideways', 'LIVE', true].map(effectiveModeOf)).toEqual([null, null, null, null, null])
  })

  it('reads the workspace switch on only when it is exactly true', () => {
    expect(workspaceSwitchOf({ proactive: { enabled: true } })).toBe(true)
    expect(workspaceSwitchOf({ proactive: { enabled: 'true' } })).toBe(false)
    expect(workspaceSwitchOf({ proactive: {} })).toBe(false)
    expect(workspaceSwitchOf({})).toBe(false)
    expect(workspaceSwitchOf(undefined)).toBe(false)
  })

  it('lets owners and admins change them, nobody else', () => {
    expect(canManageProactive('owner')).toBe(true)
    expect(canManageProactive('admin')).toBe(true)
    expect(['editor', 'viewer', 'member', undefined].map((role) => canManageProactive(role as never))).toEqual([false, false, false, false])
  })
})

describe('the caps', () => {
  it('ranges each cap no looser than the platform, and the gaps against each other', () => {
    const values = capValues(SETTINGS.caps, { min_interval_minutes: '60', max_interval_minutes: '600' })
    expect(capBounds('max_model_runs_per_day', values, SETTINGS)).toEqual({ min: 0, max: 48 })
    expect(capBounds('max_open_per_agent', values, SETTINGS)).toEqual({ min: 0, max: 3 })
    expect(capBounds('min_interval_minutes', values, SETTINGS)).toEqual({ min: 30, max: 600 })
    expect(capBounds('max_interval_minutes', values, SETTINGS)).toEqual({ min: 60, max: 1440 })
  })

  it('fits only a whole number within its range', () => {
    const fits = (name: (typeof CAP_NAMES)[number], text: string) => capFits(name, capValues(SETTINGS.caps, { [name]: text }), SETTINGS)
    expect(fits('max_model_runs_per_day', '0')).toBe(true)
    expect(fits('max_model_runs_per_day', '48')).toBe(true)
    expect(fits('max_model_runs_per_day', '49')).toBe(false)
    expect(fits('max_model_runs_per_day', '-1')).toBe(false)
    expect(fits('max_model_runs_per_day', '2.5')).toBe(false)
    expect(fits('max_model_runs_per_day', '')).toBe(false)
    expect(fits('min_interval_minutes', '29')).toBe(false)
    expect(fits('max_interval_minutes', '1441')).toBe(false)
    expect(wholeNumber(' 12 ')).toBe(12)
  })

  it('sends only what changed, and puts a cap sent as null at the platform’s', () => {
    const values = capValues(SETTINGS.caps, { max_model_runs_per_day: '12', max_open_per_agent: '1', max_open_per_person: 'x' })
    expect(changedCaps(values, SETTINGS.caps)).toEqual({ max_model_runs_per_day: 12 })

    const changed = withChange(SETTINGS, { enabled: true, caps: { max_model_runs_per_day: 12, max_open_per_agent: null } })
    expect(changed).toEqual({
      ...SETTINGS,
      enabled: true,
      caps: { ...SETTINGS.caps, max_model_runs_per_day: 12, max_open_per_agent: 3 },
      // The switch flipped: the mode is unknown until it is read again.
      effective_mode: null,
    })
    expect(withChange(SETTINGS, { caps: { max_model_runs_per_day: 12 } }).effective_mode).toBe('off')
    expect(withChange(SETTINGS, { enabled: false }).effective_mode).toBe('off')
    expect(SETTINGS.caps.max_open_per_agent).toBe(1)
    expect(DEFAULT_CAPS_CHANGE.caps).toEqual(Object.fromEntries(CAP_NAMES.map((name) => [name, null])))
  })
})
