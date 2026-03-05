// ── Legacy keys (kept for migration) ──
const LEGACY_COMPLETED_KEY = 'automatos-onboarding-completed'
const LEGACY_SKIPPED_KEY = 'automatos-onboarding-skipped'
const LEGACY_DISMISSED_KEY = 'automatos-tour-dismissed-at'
const LEGACY_COMPLETED_AT_KEY = 'automatos-tour-completed-at'

// ── New per-tour key pattern ──
const TOUR_KEY_PREFIX = 'automatos-tour'

function tourKey(tourId: string, userId: string) {
  return `${TOUR_KEY_PREFIX}:${tourId}:${userId}`
}

function legacyKey(base: string, userId: string) {
  return `${base}:${userId}`
}

// ── Per-tour API ──

export function hasSeenTour(tourId: string, userId: string): boolean {
  if (typeof window === 'undefined' || !userId) return false
  const val = localStorage.getItem(tourKey(tourId, userId))
  return val === 'completed' || val === 'skipped'
}

export function markTourComplete(tourId: string, userId: string) {
  if (typeof window === 'undefined') return
  localStorage.setItem(tourKey(tourId, userId), 'completed')
  localStorage.setItem(`${tourKey(tourId, userId)}:at`, new Date().toISOString())
}

export function markTourSkipped(tourId: string, userId: string) {
  if (typeof window === 'undefined') return
  localStorage.setItem(tourKey(tourId, userId), 'skipped')
  localStorage.setItem(`${tourKey(tourId, userId)}:at`, new Date().toISOString())
}

export function resetTour(tourId: string, userId: string) {
  if (typeof window === 'undefined') return
  localStorage.removeItem(tourKey(tourId, userId))
  localStorage.removeItem(`${tourKey(tourId, userId)}:at`)
}

export function resetAllTours(userId: string) {
  if (typeof window === 'undefined') return
  const keysToRemove: string[] = []
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key?.startsWith(`${TOUR_KEY_PREFIX}:`) && key.includes(`:${userId}`)) {
      keysToRemove.push(key)
    }
  }
  keysToRemove.forEach((k) => localStorage.removeItem(k))
}

// ── Legacy compat (used by first-login-guard & welcome-modal) ──

export function hasCompletedOnboarding(userId: string): boolean {
  if (typeof window === 'undefined' || !userId) return false
  return !!(
    localStorage.getItem(legacyKey(LEGACY_COMPLETED_KEY, userId)) ||
    localStorage.getItem(legacyKey(LEGACY_SKIPPED_KEY, userId)) ||
    hasSeenTour('welcome', userId)
  )
}

export function markOnboardingComplete(userId: string) {
  localStorage.setItem(legacyKey(LEGACY_COMPLETED_KEY, userId), 'true')
  localStorage.setItem(legacyKey(LEGACY_COMPLETED_AT_KEY, userId), new Date().toISOString())
}

export function markOnboardingSkipped(userId: string) {
  localStorage.setItem(legacyKey(LEGACY_SKIPPED_KEY, userId), 'true')
  localStorage.setItem(legacyKey(LEGACY_DISMISSED_KEY, userId), new Date().toISOString())
}

export function resetOnboarding(userId: string) {
  localStorage.removeItem(legacyKey(LEGACY_COMPLETED_KEY, userId))
  localStorage.removeItem(legacyKey(LEGACY_SKIPPED_KEY, userId))
  localStorage.removeItem(legacyKey(LEGACY_DISMISSED_KEY, userId))
  localStorage.removeItem(legacyKey(LEGACY_COMPLETED_AT_KEY, userId))
}

// ── Legacy migration ──
// Converts old single-tour keys to new per-tour format.
// Existing users who completed the old 13-step tour won't re-see welcome + agents tours.

export function migrateFromLegacy(userId: string) {
  if (typeof window === 'undefined' || !userId) return

  const alreadyMigrated = localStorage.getItem(`${TOUR_KEY_PREFIX}:migrated:${userId}`)
  if (alreadyMigrated) return

  const completed = localStorage.getItem(legacyKey(LEGACY_COMPLETED_KEY, userId))
  const skipped = localStorage.getItem(legacyKey(LEGACY_SKIPPED_KEY, userId))

  if (completed || skipped) {
    const status = completed ? 'completed' : 'skipped'
    // Mark welcome + agents tours as seen (the old tour covered both)
    markTourComplete('welcome', userId)
    markTourComplete('agents', userId)
    localStorage.setItem(`${TOUR_KEY_PREFIX}:migrated:${userId}`, 'true')
  }
}
