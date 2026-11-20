import { NextResponse, type NextRequest } from "next/server";
import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { isSaaS } from "@/lib/auth-edition";
import { isEnterprise } from "@/lib/edition-name";
import { enterpriseMiddleware } from "@/lib/enterprise/session-middleware";
import { isPublicPath, PUBLIC_ROUTES } from "@/lib/public-routes";

// PRD-175 (F008): the middleware is edition-conditional.
//   saas  → clerkMiddleware + auth.protect() on every non-public route (unchanged).
//   local → a pass-through: NO clerkMiddleware, NO auth.protect(), so EVERY route
//           is served. `local` treats the whole app as public — there is no login.
// Gated on the build-time edition flag so no Clerk symbol executes under `local`.

// The public routes are one list (lib/public-routes.ts), read here and by the enterprise session
// provider, so neither sends a public page to sign in.
const isPublicRoute = createRouteMatcher([...PUBLIC_ROUTES]);

const saasMiddleware = clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect();
  }
});

// In `local`, serve everything untouched — the app is fully public with no auth.
const localMiddleware = () => NextResponse.next();

// PRD-277 US-502: in `enterprise`, a page asked for with no session cookie goes to /sign-in and
// comes back afterwards; no clerkMiddleware, no auth.protect(). The API judges every request.
// The list is read with isPublicPath there, exactly as the session provider reads it.
const isEnterprisePublic = (request: NextRequest) => isPublicPath(request.nextUrl.pathname);

export default isSaaS ? saasMiddleware : isEnterprise ? enterpriseMiddleware(isEnterprisePublic) : localMiddleware;

export const config = {
  matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};
