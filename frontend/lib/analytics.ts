/**
 * Analytics utility for tracking user interactions
 * PRD-40: Dynamic Tool Suggestions
 *
 * Currently logs to console in development.
 * TODO: Connect to analytics service (PostHog, Mixpanel, etc.)
 */

interface EventProperties {
  [key: string]: string | number | boolean | undefined
}

export const analytics = {
  /**
   * Track an event with properties
   */
  track(eventName: string, properties?: EventProperties) {
    if (process.env.NODE_ENV === 'development') {
      console.log('[Analytics]', eventName, properties)
    }

    // TODO: Send to analytics service
    // Example: posthog.capture(eventName, properties)
  },
}
