import { NextResponse } from "next/server";
import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { isSaaS } from "@/lib/auth-edition";

// PRD-175 (F008): the middleware is edition-conditional.
//   saas  → clerkMiddleware + auth.protect() on every non-public route (unchanged).
//   local → a pass-through: NO clerkMiddleware, NO auth.protect(), so EVERY route
//           is served. `local` treats the whole app as public — there is no login.
// Gated on the build-time edition flag so no Clerk symbol executes under `local`.

const isPublicRoute = createRouteMatcher([
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
  "/verify(.*)",
]);

const saasMiddleware = clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect();
  }
});

// In `local`, serve everything untouched — the app is fully public with no auth.
const localMiddleware = () => NextResponse.next();

export default isSaaS ? saasMiddleware : localMiddleware;

export const config = {
  matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};
