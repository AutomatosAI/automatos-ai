/**
 * The web app's public routes: the one list the middleware (middleware.ts, saas and enterprise)
 * and the enterprise session provider both read, so a page served signed out is never sent to
 * sign in by either.
 *
 * Patterns are Clerk's `createRouteMatcher` patterns, kept to two shapes so `isPublicPath` reads
 * them the same way: an exact path, or a prefix followed by `(.*)`.
 */
export const PUBLIC_ROUTES: readonly string[] = [
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/reset-password(.*)",
  "/sso-callback(.*)",
  "/accept-invitation(.*)",
  "/api/webhooks(.*)",
  // PRD-261 A-08b (maintainer OK at D1-g, 18 Oct): an installed theme's files proxy. Its
  // handler authenticates (signed out → 401); auth.protect() would answer a module import()
  // or a <video> range request with a 404 before the handler runs, which reads as a broken
  // theme instead of an expired session.
  "/api/themes/files/(.*)",
  // PRD-264 EUAI-10: anyone can check a file for an Automatos AI mark, signed in or not.
  // Segment-bounded (like the theme host's isHostOwned): "/verify(.*)" would also make
  // /verify-email or /verification public.
  "/verify",
  "/verify/(.*)",
  // PRD-266 US-018c: the app-association files iOS and Android fetch signed out (Automatos Go's
  // universal links). Exact paths, never /.well-known/(.*). The matcher in middleware.ts already
  // skips assetlinks.json (it has an extension), but the extensionless AASA reaches auth.protect().
  "/.well-known/apple-app-site-association",
  "/.well-known/assetlinks.json",
  // PRD-266 US-018d: where a pairing QR opens in a phone's browser. It calls nothing and never pairs.
  // The exact path, like /verify: /go, /go/pairing and /gopher stay protected.
  "/go/pair",
  // PRD-270 US-270-08a: the Marketplace's shop window, browsed signed out. The exact path: /marketplace/publish and the rest stay protected.
  "/marketplace",
];

const ANY_REST = "(.*)";

/** Whether `path` (a pathname, no query) is one of `PUBLIC_ROUTES`, read as `createRouteMatcher` reads them. */
export function isPublicPath(path: string | null | undefined, routes: readonly string[] = PUBLIC_ROUTES): boolean {
  if (!path) return false;
  return routes.some((pattern) =>
    pattern.endsWith(ANY_REST) ? path.startsWith(pattern.slice(0, -ANY_REST.length)) : path === pattern,
  );
}
