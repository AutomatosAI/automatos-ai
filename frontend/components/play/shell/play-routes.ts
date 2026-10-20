/**
 * PRD-261 A-14f — the address of Play's own theme route, Conversations (PRD-261 A-04b). A theme's
 * route `<name>` is served at /x/<name>; Play names its route, so it names the address too and
 * imports nothing of the core for it (the SDK has no route-address builder). The core's
 * registration (lib/theme-host/builtin/play-entry.ts, PLAY_CONVERSATIONS_HREF) says the same, and
 * lib/__tests__/prd261-port-frame.test.ts holds the two equal.
 */

/** Where every theme route is served (app/x/[...path]/page.tsx). */
const THEME_ROUTE_PREFIX = '/x/'

/** Play's Conversations page: its theme route `conversations`. */
export const PLAY_CONVERSATIONS_ROUTE = 'conversations'
export const PLAY_CONVERSATIONS_HREF = `${THEME_ROUTE_PREFIX}${PLAY_CONVERSATIONS_ROUTE}`
