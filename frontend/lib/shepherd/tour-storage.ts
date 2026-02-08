const TOUR_COMPLETED_KEY = 'automatos-onboarding-completed'
const TOUR_SKIPPED_KEY = 'automatos-onboarding-skipped'
const TOUR_DISMISSED_KEY = 'automatos-tour-dismissed-at'
const TOUR_COMPLETED_AT_KEY = 'automatos-tour-completed-at'

function userKey(base: string, userId: string) {
  return `${base}:${userId}`
}

export function hasCompletedOnboarding(userId: string): boolean {
  if (typeof window === 'undefined' || !userId) return false

  return !!(
    localStorage.getItem(userKey(TOUR_COMPLETED_KEY, userId)) ||
    localStorage.getItem(userKey(TOUR_SKIPPED_KEY, userId))
  )
}

export function markOnboardingComplete(userId: string) {
  localStorage.setItem(userKey(TOUR_COMPLETED_KEY, userId), 'true')
  localStorage.setItem(userKey(TOUR_COMPLETED_AT_KEY, userId), new Date().toISOString())
}

export function markOnboardingSkipped(userId: string) {
  localStorage.setItem(userKey(TOUR_SKIPPED_KEY, userId), 'true')
  localStorage.setItem(userKey(TOUR_DISMISSED_KEY, userId), new Date().toISOString())
}

export function resetOnboarding(userId: string) {
  localStorage.removeItem(userKey(TOUR_COMPLETED_KEY, userId))
  localStorage.removeItem(userKey(TOUR_SKIPPED_KEY, userId))
  localStorage.removeItem(userKey(TOUR_DISMISSED_KEY, userId))
  localStorage.removeItem(userKey(TOUR_COMPLETED_AT_KEY, userId))
}
