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
