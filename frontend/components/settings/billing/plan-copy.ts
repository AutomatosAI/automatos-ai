/**
 * PRD-265 US-011 — the plan picker's words, from the public facts sheet (§4 standard block,
 * §5 Enterprise card), word for word, so the in-app upgrade surface and the Studio site say
 * the same thing.
 *
 * While billing is closed (`billing_open` false) the beta sentences show and no plan shows a
 * credit amount; once it opens the beta sentences go and Pro and Business show their
 * `monthly_credits`. Plan names and prices always come from the API (`name`, `price_usd`).
 *
 * PRD-265 W2: Business is by invite. While a platform admin's offer to this workspace stands
 * (`business_offer_until` in the future) its card can be bought here and says until when;
 * otherwise it offers "Contact Automatos", as Enterprise always does.
 */
import { isStillOpen, type BillingPlan, type BillingPlans } from '@/hooks/use-billing-plans'
import { formatCredits, formatDay, formatUsd } from './billing-format'

export const BUSINESS_PLAN = 'business'
export const CONTACT_URL = 'https://automatos.app/contact'
export const CONTACT_LABEL = 'Contact Automatos'
export const COMING_SOON = 'Coming soon'
export const OWNER_ONLY_NOTE = 'Only a workspace owner or admin can change the plan or buy credits.'
export const CHECKOUT_CLOSED_NOTE = "Checkout isn't available right now. Try again later."

/** Facts sheet §4 (Own keys, Pro, Business) and §5 (Enterprise card), by plan key. */
const PLAN_BODIES: Readonly<Record<string, string>> = {
  basic: 'Everything in Pro, on your own Anthropic, OpenAI or OpenRouter keys. Your provider bills you for the AI.',
  pro: "5 seats, 20 agents, 60 minutes of video render a month, with monthly AI credits included, so you don't need keys of your own.",
  business: '25 seats, unlimited agents, 240 minutes of video render a month, voice, and a larger monthly credit allowance.',
  enterprise: "The same OS in your own cluster, with a licence, committed credits and consultancy. We're building it now. Tell us what you need.",
}
const INVITE_BETA = 'By invite during the beta: contact Automatos.'
const INVITE_OPEN = 'By invite: contact Automatos.'

const BETA_SENTENCES =
  'Beta pricing: not charged until billing opens. During the beta every plan runs on the trial and then on your own AI keys; monthly credits on Pro and Business start when billing opens.'
const LOCAL_EDITION_SENTENCE =
  'The local edition is free and open source (Apache 2.0), on your own keys or your Claude Code / CLI subscription.'
const DEFAULT_TRIAL_CREDITS = 500

/**
 * Not sold here, or Business without an open offer: the card offers "Contact Automatos" and
 * never a payment form.
 */
export function isContactOnly(plan: BillingPlan, offerUntil: string | null | undefined = null): boolean {
  if (plan.coming_soon || !plan.self_serve) return true
  return plan.key === BUSINESS_PLAN && !isStillOpen(offerUntil)
}

/** "Offered to you until 6 Nov 2026." on Business while its offer stands; null otherwise. */
export function planOfferLine(plan: BillingPlan, offerUntil: string | null | undefined): string | null {
  if (plan.key !== BUSINESS_PLAN || isContactOnly(plan, offerUntil)) return null
  return `Offered to you until ${formatDay(offerUntil)}.`
}

/** "$29 a month"; "Coming soon" for a plan not on sale yet. */
export function planPriceLabel(plan: BillingPlan): string {
  if (plan.coming_soon || plan.price_usd === null) return COMING_SOON
  return `${formatUsd(plan.price_usd)} a month`
}

/** The plan's body copy, with the invitation sentence on an invite-only plan nobody has offered. */
export function planDescription(plan: BillingPlan, billingOpen: boolean, contactOnly: boolean = isContactOnly(plan)): string {
  const body = PLAN_BODIES[plan.key] ?? ''
  if (plan.coming_soon || !contactOnly) return body
  return [body, billingOpen ? INVITE_OPEN : INVITE_BETA].filter(Boolean).join(' ')
}

/** "2,000 credits a month included": only once billing is open, and only on a plan with credits. */
export function planCreditsLine(plan: BillingPlan, billingOpen: boolean): string | null {
  if (!billingOpen || !plan.monthly_credits || plan.monthly_credits <= 0) return null
  return `${formatCredits(plan.monthly_credits)} credits a month included.`
}

/** The paragraph under the plans: the trial, the credit unit, the beta terms, the local edition. */
export function plansFootnote(billingOpen: boolean, trialCredits: number | null | undefined): string {
  const trial = formatCredits(trialCredits ?? DEFAULT_TRIAL_CREDITS)
  const sentences = [
    `New accounts start with a one-time ${trial}-credit trial.`,
    'A credit is one US cent of AI usage at Automatos prices.',
    billingOpen ? null : BETA_SENTENCES,
    LOCAL_EDITION_SENTENCE,
  ]
  return sentences.filter(Boolean).join(' ')
}

/** Why there are no checkout buttons, when the copy above does not already say. */
export function pickerNote(plans: BillingPlans): string | null {
  if (!plans.billing_open) return null
  if (!plans.can_manage) return OWNER_ONLY_NOTE
  return plans.checkout_available ? null : CHECKOUT_CLOSED_NOTE
}
