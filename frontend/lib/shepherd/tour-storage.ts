const TOUR_COMPLETED_KEY = 'automatos-onboarding-completed'
const TOUR_SKIPPED_KEY = 'automatos-onboarding-skipped'
const TOUR_DISMISSED_KEY = 'automatos-tour-dismissed-at'

export function hasCompletedOnboarding(): boolean {
  if (typeof window === 'undefined') return false

  return !!(
    localStorage.getItem(TOUR_COMPLETED_KEY) ||
    localStorage.getItem(TOUR_SKIPPED_KEY)
  )
}

export function markOnboardingComplete() {
  localStorage.setItem(TOUR_COMPLETED_KEY, 'true')
  localStorage.setItem('automatos-tour-completed-at', new Date().toISOString())
}

export function markOnboardingSkipped() {
  localStorage.setItem(TOUR_SKIPPED_KEY, 'true')
  localStorage.setItem(TOUR_DISMISSED_KEY, new Date().toISOString())
}

export function resetOnboarding() {
  localStorage.removeItem(TOUR_COMPLETED_KEY)
  localStorage.removeItem(TOUR_SKIPPED_KEY)
  localStorage.removeItem(TOUR_DISMISSED_KEY)
  localStorage.removeItem('automatos-tour-completed-at')
}
