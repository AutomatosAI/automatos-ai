/**
 * Tour Registry — maps routes to tour modules.
 * All tour factories are lazy-loaded to keep the bundle SSR-safe.
 */

export interface TourRegistryEntry {
  /** Unique tour ID, matches key in tour-storage */
  id: string
  /** Human-readable label for Guide button menu */
  label: string
  /** Lazy factory that returns a Shepherd tour instance */
  factory: (userId: string) => Promise<any>
}

const registry: Record<string, TourRegistryEntry> = {
  '/chat': {
    id: 'chat',
    label: 'Chat',
    factory: async (userId) => {
      const { createChatTour } = await import('./tours/chat-tour')
      return createChatTour(userId)
    },
  },
  '/marketplace': {
    id: 'marketplace',
    label: 'Marketplace',
    factory: async (userId) => {
      const { createMarketplaceTour } = await import('./tours/marketplace-tour')
      return createMarketplaceTour(userId)
    },
  },
  '/agents': {
    id: 'agents',
    label: 'Agents',
    factory: async (userId) => {
      const { createAgentsTour } = await import('./tours/agents-tour')
      return createAgentsTour(userId)
    },
  },
  '/tools': {
    id: 'tools',
    label: 'Tools & Integrations',
    factory: async (userId) => {
      const { createToolsTour } = await import('./tours/tools-tour')
      return createToolsTour(userId)
    },
  },
  '/documents': {
    id: 'documents',
    label: 'Knowledge Bases',
    factory: async (userId) => {
      const { createDocumentsTour } = await import('./tours/documents-tour')
      return createDocumentsTour(userId)
    },
  },
  '/workflows': {
    id: 'workflows',
    label: 'Workflows',
    factory: async (userId) => {
      const { createWorkflowsTour } = await import('./tours/workflows-tour')
      return createWorkflowsTour(userId)
    },
  },
  '/analytics': {
    id: 'analytics',
    label: 'Analytics',
    factory: async (userId) => {
      const { createAnalyticsTour } = await import('./tours/analytics-tour')
      return createAnalyticsTour(userId)
    },
  },
  '/settings': {
    id: 'settings',
    label: 'Settings',
    factory: async (userId) => {
      const { createSettingsTour } = await import('./tours/settings-tour')
      return createSettingsTour(userId)
    },
  },
}

/** Get the tour entry for a given pathname, or undefined if no tour exists */
export function getTourForRoute(pathname: string): TourRegistryEntry | undefined {
  // Exact match first
  if (registry[pathname]) return registry[pathname]
  // Prefix match for nested routes (e.g. /chat/abc → /chat)
  for (const route of Object.keys(registry)) {
    if (pathname.startsWith(route)) return registry[route]
  }
  return undefined
}

/** Get all registered tours (for Guide button "Reset All" and menu) */
export function getAllTours(): TourRegistryEntry[] {
  return Object.values(registry)
}

/** Get the welcome tour factory (special — not route-based) */
export async function createWelcomeTour(userId: string) {
  const { createWelcomeTour: factory } = await import('./tours/welcome-tour')
  return factory(userId)
}
