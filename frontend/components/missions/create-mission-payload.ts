/**
 * The pure half of the New mission form's submit: the goal sentence, the optional config and the
 * create request, built from what the form holds. Nothing here renders, fetches or toasts.
 */

import type { MissionCreateRequest } from '@/types/missions'
import {
  BUSINESS_PLAN_TEMPLATE_ID,
  DEFAULT_POWER_MODE,
  type MissionAttachment,
  type PowerMode,
} from './create-mission-options'

export const BUSINESS_PLAN_REQUIRED_MESSAGE = 'Business name, type, and industry are required'
export const GOAL_REQUIRED_MESSAGE = 'Please enter a mission name or description'

export interface MissionFormValues {
  selectedTemplate: string | null
  name: string
  description: string
  tags: string
  budgetPauseEnabled: boolean
  powerMode: PowerMode
  /** F282: "Check each step with me" */
  checkEachStep: boolean
}

/** The Business Plan template's extra fields. */
export interface BusinessPlanValues {
  businessName: string
  businessType: string
  industry: string
  targetMarket: string
  businessGoals: string
}

export type MissionGoalResult = { goal: string } | { error: string }

export function isBusinessPlanTemplate(templateId: string | null): boolean {
  return templateId === BUSINESS_PLAN_TEMPLATE_ID
}

/** The Business Plan template's three required fields are filled in. Pure. */
export function businessPlanComplete(plan: BusinessPlanValues): boolean {
  return Boolean(plan.businessName.trim() && plan.businessType.trim() && plan.industry.trim())
}

// For business plan template, build goal from structured fields
function businessPlanGoal(plan: BusinessPlanValues, description: string): string {
  const parts = [
    `Write a business plan for ${plan.businessName.trim()}`,
    `a ${plan.businessType.trim()} business in the ${plan.industry.trim()} industry`,
  ]
  if (plan.targetMarket.trim()) parts.push(`targeting ${plan.targetMarket.trim()}`)
  if (plan.businessGoals.trim()) parts.push(`with goals: ${plan.businessGoals.trim()}`)
  if (description.trim()) parts.push(description.trim())
  return parts.join('. ')
}

function customGoal(name: string, description: string): string {
  const goalParts: string[] = []
  if (name.trim()) goalParts.push(name.trim())
  if (description.trim()) goalParts.push(description.trim())
  return goalParts.join(': ')
}

/** The mission's goal sentence, or the message to show when the form cannot make one. Pure. */
export function missionGoal(form: MissionFormValues, plan: BusinessPlanValues): MissionGoalResult {
  let goal: string
  if (isBusinessPlanTemplate(form.selectedTemplate)) {
    if (!businessPlanComplete(plan)) return { error: BUSINESS_PLAN_REQUIRED_MESSAGE }
    goal = businessPlanGoal(plan, form.description)
  } else {
    goal = customGoal(form.name, form.description)
  }
  return goal ? { goal } : { error: GOAL_REQUIRED_MESSAGE }
}

// Add business plan fields to config for downstream agents
function businessPlanConfig(plan: BusinessPlanValues): Record<string, unknown> {
  return {
    business_name: plan.businessName.trim(),
    business_type: plan.businessType.trim(),
    industry: plan.industry.trim(),
    ...(plan.targetMarket.trim() ? { target_market: plan.targetMarket.trim() } : {}),
    ...(plan.businessGoals.trim() ? { goals: plan.businessGoals.trim() } : {}),
  }
}

/** The mission config from the form's optional fields; a field left at its default sends nothing. Pure. */
export function missionConfig(
  form: MissionFormValues,
  plan: BusinessPlanValues,
  attachments: MissionAttachment[],
): Record<string, unknown> {
  const tagList = form.tags
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
  return {
    ...(tagList.length > 0 ? { tags: tagList } : {}),
    ...(form.name.trim() ? { name: form.name.trim() } : {}),
    // PRD-127: Send attachment_ids (list of UUID strings) instead of document refs
    ...(attachments.length > 0 ? { attachment_ids: attachments.map((a) => a.attachment_id) } : {}),
    ...(!form.budgetPauseEnabled ? { budget_pause_disabled: true } : {}),
    ...(form.powerMode !== DEFAULT_POWER_MODE ? { power_mode: form.powerMode } : {}),
    ...(form.checkEachStep ? { check_each_step: true } : {}),
    ...(isBusinessPlanTemplate(form.selectedTemplate) ? businessPlanConfig(plan) : {}),
  }
}

/** The create request: the config and the template only when there is one. Pure. */
export function createMissionRequest(
  goal: string,
  config: Record<string, unknown>,
  templateId: string | null,
): MissionCreateRequest {
  return {
    goal,
    ...(Object.keys(config).length > 0 ? { config } : {}),
    ...(templateId ? { template_id: templateId } : {}),
  }
}
