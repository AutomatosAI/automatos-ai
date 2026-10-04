/**
 * The New mission form's request, built by create-mission-payload.ts since the modal was split
 * (F282 shape gate): the same goal, config and template the form sent before.
 */
import { describe, it, expect } from 'vitest'
import {
  BUSINESS_PLAN_REQUIRED_MESSAGE,
  GOAL_REQUIRED_MESSAGE,
  createMissionRequest,
  missionConfig,
  missionGoal,
  type BusinessPlanValues,
  type MissionFormValues,
} from '../create-mission-payload'

const FORM: MissionFormValues = {
  selectedTemplate: null,
  name: '',
  description: '',
  tags: '',
  budgetPauseEnabled: true,
  powerMode: 'standard',
  checkEachStep: false,
}

const NO_PLAN: BusinessPlanValues = {
  businessName: '',
  businessType: '',
  industry: '',
  targetMarket: '',
  businessGoals: '',
}

const PLAN: BusinessPlanValues = {
  businessName: ' BrewCraft ',
  businessType: 'retail',
  industry: 'Coffee',
  targetMarket: 'Urban millennials',
  businessGoals: '',
}

describe('missionGoal', () => {
  it('joins the name and description of a custom goal', () => {
    expect(missionGoal({ ...FORM, name: ' Ship it ', description: 'by Friday' }, NO_PLAN)).toEqual({ goal: 'Ship it: by Friday' })
  })

  it('asks for a name or description when both are empty', () => {
    expect(missionGoal(FORM, NO_PLAN)).toEqual({ error: GOAL_REQUIRED_MESSAGE })
  })

  it('needs the business plan required fields', () => {
    expect(missionGoal({ ...FORM, selectedTemplate: 'business_plan' }, NO_PLAN)).toEqual({ error: BUSINESS_PLAN_REQUIRED_MESSAGE })
  })

  it('writes the business plan goal from its fields', () => {
    expect(missionGoal({ ...FORM, selectedTemplate: 'business_plan', description: 'Keep it short' }, PLAN)).toEqual({
      goal: 'Write a business plan for BrewCraft. a retail business in the Coffee industry. targeting Urban millennials. Keep it short',
    })
  })
})

describe('missionConfig', () => {
  it('sends nothing for fields left at their defaults', () => {
    expect(missionConfig(FORM, NO_PLAN, [])).toEqual({})
  })

  it('sends each changed option under its own key', () => {
    const form = { ...FORM, name: 'N', tags: 'a, ,b', budgetPauseEnabled: false, powerMode: 'max' as const, checkEachStep: true }
    const attachment = { attachment_id: 'att-1', filename: 'f.pdf', mime: 'application/pdf', media_type: 'document' as const }
    expect(missionConfig(form, NO_PLAN, [attachment])).toEqual({
      tags: ['a', 'b'],
      name: 'N',
      attachment_ids: ['att-1'],
      budget_pause_disabled: true,
      power_mode: 'max',
      check_each_step: true,
    })
  })

  it('adds the business plan fields for that template only', () => {
    expect(missionConfig({ ...FORM, selectedTemplate: 'business_plan' }, PLAN, [])).toEqual({
      business_name: 'BrewCraft',
      business_type: 'retail',
      industry: 'Coffee',
      target_market: 'Urban millennials',
    })
    expect(missionConfig(FORM, PLAN, [])).toEqual({})
  })
})

describe('createMissionRequest', () => {
  it('leaves out an empty config and a missing template', () => {
    expect(createMissionRequest('g', {}, null)).toEqual({ goal: 'g' })
    expect(createMissionRequest('g', { name: 'n' }, 'business_plan')).toEqual({
      goal: 'g',
      config: { name: 'n' },
      template_id: 'business_plan',
    })
  })
})
