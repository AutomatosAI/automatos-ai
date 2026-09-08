/**
 * The Create Task dialog's "When" row: `now` keeps the existing create; every
 * other mode becomes a scheduled task. Crons are UTC (the scheduler's clock).
 */
import { describe, it, expect } from 'vitest'
import {
  buildSchedulePayload,
  defaultScheduleAt,
  describeSchedule,
  isScheduleInFuture,
} from '../schedule-choice'

const LOCAL = '2030-01-15T09:30' // a Tuesday, well in the future
const at = new Date(LOCAL)

describe('buildSchedulePayload', () => {
  it('now → null (the existing create path)', () => {
    expect(buildSchedulePayload('now', LOCAL)).toBeNull()
  })

  it('later → a one-shot at the chosen instant, in ISO/UTC', () => {
    expect(buildSchedulePayload('later', LOCAL)).toEqual({
      task_type: 'one_shot',
      schedule: at.toISOString(),
    })
  })

  it('daily / weekdays / weekly → 5-field crons from the UTC parts of the time', () => {
    const m = at.getUTCMinutes()
    const h = at.getUTCHours()
    expect(buildSchedulePayload('daily', LOCAL)).toEqual({ task_type: 'recurring', schedule: `${m} ${h} * * *` })
    expect(buildSchedulePayload('weekdays', LOCAL)).toEqual({ task_type: 'recurring', schedule: `${m} ${h} * * 1-5` })
    expect(buildSchedulePayload('weekly', LOCAL)).toEqual({
      task_type: 'recurring',
      schedule: `${m} ${h} * * ${at.getUTCDay()}`,
    })
  })

  it('an unparseable time → null, never a broken cron', () => {
    expect(buildSchedulePayload('later', '')).toBeNull()
    expect(buildSchedulePayload('daily', 'not a date')).toBeNull()
  })
})

describe('helpers', () => {
  it('defaultScheduleAt is tomorrow 09:00 local in datetime-local form', () => {
    const v = defaultScheduleAt(new Date('2026-09-08T20:15:00'))
    expect(v).toBe('2026-09-09T09:00')
  })

  it('isScheduleInFuture compares against now', () => {
    expect(isScheduleInFuture(LOCAL, new Date('2026-01-01T00:00:00'))).toBe(true)
    expect(isScheduleInFuture('2020-01-01T09:00', new Date('2026-01-01T00:00:00'))).toBe(false)
    expect(isScheduleInFuture('')).toBe(false)
  })

  it('describeSchedule reads naturally for each mode', () => {
    expect(describeSchedule('later', LOCAL)).toMatch(/^for Tue 15 Jan, 09:30$/)
    expect(describeSchedule('daily', LOCAL)).toBe('every day at 09:30')
    expect(describeSchedule('weekdays', LOCAL)).toBe('every weekday at 09:30')
    expect(describeSchedule('weekly', LOCAL)).toBe('every Tuesday at 09:30')
    expect(describeSchedule('later', '')).toBe('')
  })
})
