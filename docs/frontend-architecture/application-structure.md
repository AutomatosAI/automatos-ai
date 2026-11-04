# Application Structure

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [charts/automatos/templates/frontend.yaml](charts/automatos/templates/frontend.yaml)
- [charts/automatos/tests/otel_test.yaml](charts/automatos/tests/otel_test.yaml)
- [frontend/app/__tests__/studio-element-rules-contract.test.ts](frontend/app/__tests__/studio-element-rules-contract.test.ts)
- [frontend/app/sso-callback/page.tsx](frontend/app/sso-callback/page.tsx)
- [frontend/components/admin/themes/signature-panel.tsx](frontend/components/admin/themes/signature-panel.tsx)
- [frontend/components/marketplace/themes/__tests__/listing-fixtures.ts](frontend/components/marketplace/themes/__tests__/listing-fixtures.ts)
- [frontend/components/theme-host/__tests__/asleep-slot.test.tsx](frontend/components/theme-host/__tests__/asleep-slot.test.tsx)
- [frontend/components/theme-host/__tests__/theme-frame-below-min-width.test.tsx](frontend/components/theme-host/__tests__/theme-frame-below-min-width.test.tsx)
- [frontend/csp-sources.js](frontend/csp-sources.js)
- [frontend/instrumentation.ts](frontend/instrumentation.ts)
- [frontend/lib/__tests__/clerk-production-csp.test.ts](frontend/lib/__tests__/clerk-production-csp.test.ts)
- [frontend/lib/__tests__/f352-pdf-preview-frame.test.ts](frontend/lib/__tests__/f352-pdf-preview-frame.test.ts)
- [frontend/lib/__tests__/prd256-o6b-otel-fetch.test.ts](frontend/lib/__tests__/prd256-o6b-otel-fetch.test.ts)
- [frontend/lib/__tests__/prd256-o6b-otel.test.ts](frontend/lib/__tests__/prd256-o6b-otel.test.ts)
- [frontend/lib/__tests__/stripe-csp.test.ts](frontend/lib/__tests__/stripe-csp.test.ts)
- [frontend/lib/otel/node.ts](frontend/lib/otel/node.ts)
- [frontend/lib/otel/settings.ts](frontend/lib/otel/settings.ts)
- [frontend/next.config.js](frontend/next.config.js)
- [frontend/package-lock.json](frontend/package-lock.json)
- [frontend/package.json](frontend/package.json)
- [orchestrator/api/admin_credits.py](orchestrator/api/admin_credits.py)
- [orchestrator/services/credits/alerts.py](orchestrator/services/credits/alerts.py)
- [orchestrator/services/credits/meter.py](orchestrator/services/credits/meter.py)
- [orchestrator/services/credits/routing.py](orchestrator/services/credits/routing.py)
- [orchestrator/services/credits/summary.py](orchestrator/services/credits/summary.py)
- [orchestrator/tests/test_prd265_billing_api.py](orchestrator/tests/test_prd265_billing_api.py)
- [orchestrator/tests/test_prd265_w2_admin_business_invite.py](orchestrator/tests/test_prd265_w2_admin_business_invite.py)

</details>



## Purpose and Scope

This page describes the frontend application structure: package tooling, Next.js configuration and response security policies, root layout and provider hierarchy, and styling and theme selection. The available file contents establish the package manifest, CSP construction, and OpenTelemetry implementation; the existing page’s layout and provider descriptions are retained as documented there. [frontend/package.json:1-132](), [frontend/next.config.js:1-123](), [frontend/csp-sources.js:1-117](), [frontend/lib/otel/node.ts:1-129]()

Sources: [frontend/package.json:1-132](), [frontend/next.config.js:1-123](), [frontend/csp-sources.js:1-117](), [frontend/lib/otel/node.ts:1-129]()

---

## Framework, Runtime, and Package Tooling

The frontend package is named `app`, is private, and requires Node.js `>=20.0.0`. Its scripts coordinate development, production builds, linting, and Vitest testing. The `predev` and `prebuild` hooks generate theme SDK shims before their corresponding Next.js commands. [frontend/package.json:1-16]()

| Script or setting | Role |
|---|---|
| `predev` | Runs `node scripts/theme-sdk/generate-shims.js` before `next dev`. [frontend/package.json:4-6]() |
| `dev` | Starts the Next.js development server. [frontend/package.json:4-6]() |
| `prebuild` | Generates theme SDK shims before the production build. [frontend/package.json:7-8]() |
| `build` | Compiles the production application with `next build`. [frontend/package.json:7-8]() |
| `start` | Launches the production server with `next start`. [frontend/package.json:9]() |
| `lint` | Runs `next lint`. [frontend/package.json:10]() |
| `test` / `test:watch` | Run Vitest once or in watch mode. [frontend/package.json:11-12]() |
| `engines.node` | Declares Node.js `>=20.0.0`. [frontend/package.json:14-16]() |

The manifest declares Next.js `^15.5.28`, React and React DOM `18.2.0`, Clerk’s Next.js integration, TanStack React Query, Zustand, and the AI SDK packages. It also includes OpenTelemetry packages for Node-side tracing. [frontend/package.json:45-125]()

### Natural Language to Code Entity Map: Package Tooling

**Diagram: Frontend Tooling Entities**

```mermaid
flowchart LR
    "Developer starts app" --> "predev"
    "predev" --> "scripts/theme-sdk/generate-shims.js"
    "Developer starts app" --> "dev"
    "dev" --> "next dev"
    "Developer builds app" --> "prebuild"
    "prebuild" --> "scripts/theme-sdk/generate-shims.js"
    "Developer builds app" --> "build"
    "build" --> "next build"
    "Production server" --> "start"
    "start" --> "next start"
    "Server state management" --> "@tanstack/react-query"
    "Client state store" --> "zustand"
    "Authentication integration" --> "@clerk/nextjs"
    "AI SDK integration" --> "@ai-sdk/react"
```

Sources: [frontend/package.json:4-16](), [frontend/package.json:45-125]()

---

## Next.js Configuration and Response Security

`frontend/next.config.js` builds security headers and the Content-Security-Policy (CSP) for all routes. It derives `apiOrigin` from `NEXT_PUBLIC_API_URL`, media origins from `NEXT_PUBLIC_MEDIA_ORIGINS`, Stripe sources from `NEXT_PUBLIC_AUTH_EDITION`, and the Clerk Frontend API origin from `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`. These values are composed into `SECURITY_HEADERS_RULE`. [frontend/next.config.js:11-36]()

`SECURITY_HEADERS_RULE` applies `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, `Strict-Transport-Security`, `X-DNS-Prefetch-Control`, and a CSP to the `/(.*)` route pattern. CSP directives include script, style, image, media, font, connection, frame, worker, object, base, form, and frame-ancestor policies. [frontend/next.config.js:35-123]()

| Configuration entity | Implementation detail |
|---|---|
| `apiOrigin` | Parses the origin from `NEXT_PUBLIC_API_URL`; invalid or absent values yield an empty string. [frontend/next.config.js:12-19]() |
| `mediaSources` | Calls `mediaOrigins()` with `NEXT_PUBLIC_MEDIA_ORIGINS` and joins the validated origins for CSP directives. [frontend/next.config.js:21-22](), [frontend/csp-sources.js:19-26]() |
| `stripeCsp` | Calls `stripeCspFor()` using `NEXT_PUBLIC_AUTH_EDITION`; the local edition contributes no Stripe sources. [frontend/next.config.js:24-25](), [frontend/csp-sources.js:80-87]() |
| `clerkFrontendApi` | Derives the Clerk Frontend API origin from the publishable key. [frontend/next.config.js:27-30](), [frontend/csp-sources.js:39-55]() |
| `SECURITY_HEADERS_RULE` | Defines headers for all routes, including the composed CSP. [frontend/next.config.js:35-123]() |

`mediaOrigins()` accepts comma- or whitespace-separated tokens but keeps only bare HTTP(S) origins matching its origin pattern. `clerkFrontendApiOrigin()` validates the publishable key and decoded host before returning an origin. `cloudImageSources()` limits Cloud catalogue image permissions to the catalogue files path rather than allowing an unrestricted origin. [frontend/csp-sources.js:19-26](), [frontend/csp-sources.js:39-55](), [frontend/csp-sources.js:89-105]()

### Natural Language to Code Entity Map: CSP Construction

**Diagram: Build-Time CSP Source Flow**

```mermaid
flowchart TD
    "NEXT_PUBLIC_API_URL" --> "apiOrigin"
    "NEXT_PUBLIC_MEDIA_ORIGINS" --> "mediaOrigins"
    "mediaOrigins" --> "mediaSources"
    "NEXT_PUBLIC_AUTH_EDITION" --> "stripeCspFor"
    "stripeCspFor" --> "stripeCsp"
    "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" --> "clerkFrontendApiOrigin"
    "clerkFrontendApiOrigin" --> "clerkFrontendApi"
    "apiOrigin" --> "SECURITY_HEADERS_RULE"
    "mediaSources" --> "SECURITY_HEADERS_RULE"
    "stripeCsp" --> "SECURITY_HEADERS_RULE"
    "clerkFrontendApi" --> "SECURITY_HEADERS_RULE"
    "SECURITY_HEADERS_RULE" --> "Content-Security-Policy"
```

Sources: [frontend/next.config.js:11-123](), [frontend/csp-sources.js:19-26](), [frontend/csp-sources.js:39-55](), [frontend/csp-sources.js:80-105]()

---

## Root Layout Hierarchy and Providers

The root layout and provider hierarchy described here establish the document shell and application-wide context boundaries. The root layout resolves UI-style and theme boot configuration before rendering the document; the provider tree then composes authentication, role, theme, query, workspace, and theme-host contexts around the application. [frontend/app/layout.tsx:33-73](), [frontend/components/providers.tsx:33-141]()

The `<html>` element includes font variable classes and hydration-warning suppression, while the `<head>` includes theme import-map and boot-script components. The body renders `Providers` around the application content. [frontend/app/layout.tsx:45-73]()

**Diagram: Root Layout and Provider Composition**

```mermaid
flowchart TD
    "RootLayout" --> "Providers"
    "Providers" --> "AuthBoundary"
    "AuthBoundary" --> "RoleProvider"
    "RoleProvider" --> "ThemeProvider"
    "ThemeProvider" --> "UiStyleProvider"
    "UiStyleProvider" --> "QueryClientProvider"
    "QueryClientProvider" --> "WorkspaceProvider"
    "WorkspaceProvider" --> "ChooseInstallProvider"
    "ChooseInstallProvider" --> "ThemeHostProvider"
    "ThemeHostProvider" --> "ThemeLookRuntime"
    "ThemeLookRuntime" --> "GlobalSearch"
    "ThemeLookRuntime" --> "BriefOnOpen"
    "ThemeLookRuntime" --> "children"
```

Sources: [frontend/app/layout.tsx:33-73](), [frontend/components/providers.tsx:33-141]()

`AuthBoundary` selects the authentication provider according to the edition: SaaS mode renders Clerk providers, while local mode renders `LocalAuthProvider`. [frontend/components/providers.tsx:28-79]()

Sources: [frontend/app/layout.tsx:33-73](), [frontend/components/providers.tsx:28-79]()

---

## Middleware and Route-Level Behavior

The provided file contents do not include a middleware implementation or a complete App Router route inventory, so this page does not infer middleware matching, redirects, or route-group behavior. One route-level example is `frontend/app/sso-callback/page.tsx`: `SSOCallbackPage` checks whether `/sso-callback` is available in the current edition, redirects to `/` when unavailable, and otherwise renders Clerk’s `AuthenticateWithRedirectCallback` with the home route as its sign-in and sign-up destination. [frontend/app/sso-callback/page.tsx:1-19]()

Sources: [frontend/app/sso-callback/page.tsx:1-19]()

---

## Server-Side OpenTelemetry

The frontend OpenTelemetry settings are read by `frontend/instrumentation.ts` before loading the tracing implementation; the implementation itself is Node-runtime only. When started, `startTracing()` installs a `NodeTracerProvider`, registers it globally, and enables Undici instrumentation so Node `fetch` requests can propagate trace context. The provider uses a parent-based sampler with a configurable trace-ratio sampler and batches spans through a redacting exporter wrapping OTLP/HTTP. [frontend/lib/otel/settings.ts:1-5](), [frontend/lib/otel/node.ts:1-14](), [frontend/lib/otel/node.ts:94-122]()

`buildProvider()` names the service from `OTEL_SERVICE_NAME` or the default `automatos-web`, attaches the edition resource attribute, merges detected environment resources, and configures sampling and export. `startTracing()` is idempotent after successful provider installation and returns `false` if initialization fails. [frontend/lib/otel/node.ts:94-122]()

Before export, `RedactingExporter` applies `redactedSpan()` to each span. URL query values are replaced with `REDACTED`; exception events retain only approved type-related attributes, and error status text is constrained to an error type. A span that cannot be redacted is dropped rather than exported unchanged. [frontend/lib/otel/node.ts:35-92](), [frontend/lib/otel/settings.ts:7-56]()

### Natural Language to Code Entity Map: Web Tracing

**Diagram: Request Tracing and Redaction Entities**

```mermaid
flowchart LR
    "OTEL_ENABLED" --> "otelEnabled"
    "instrumentation.ts" --> "startTracing"
    "startTracing" --> "buildProvider"
    "buildProvider" --> "NodeTracerProvider"
    "NodeTracerProvider" --> "UndiciInstrumentation"
    "NodeTracerProvider" --> "BatchSpanProcessor"
    "BatchSpanProcessor" --> "RedactingExporter"
    "RedactingExporter" --> "redactedSpan"
    "redactedSpan" --> "redactQuery"
    "redactedSpan" --> "redactedAttributes"
    "RedactingExporter" --> "OTLPTraceExporter"
    "Node fetch to API" --> "traceparent"
```

Sources: [frontend/lib/otel/settings.ts:7-29](), [frontend/lib/otel/node.ts:54-103](), [frontend/lib/otel/node.ts:106-122]()

---

## Styling and Theme Toggle Subsystem

Tailwind CSS is configured with custom font families, animations, and CSS-variable-backed color tokens. Dark mode uses the `.dark` class strategy. [frontend/tailwind.config.ts:1-172]()

The existing theme-toggle implementation provides a style axis for `classic`, `studio`, and workspace-installed themes, and a tone axis for `light`, `dark`, and `system`. Theme policies can lock the tone selection. [frontend/components/ui/theme-toggle.tsx:80-92](), [frontend/components/ui/theme-toggle.tsx:99-163]()

Sources: [frontend/tailwind.config.ts:1-172](), [frontend/components/ui/theme-toggle.tsx:80-163](), [frontend/lib/theme-host/choice-sync.ts:1-120]()

---