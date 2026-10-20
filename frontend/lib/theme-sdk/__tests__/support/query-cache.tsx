/**
 * PRD-261 W-A4P — a react-query client a page-area test can look into, to prove a theme's view
 * never hands on the objects the host caches for the core page (internal/query-view.ts).
 */
import type { ReactElement, ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const SILENT = { log: () => undefined, warn: () => undefined, error: () => undefined }

export interface InspectableClient {
  readonly client: QueryClient
  readonly wrapper: (props: { children: ReactNode }) => ReactElement
}

/** A fresh client with no retries, and the provider that hands it to the hooks under test. */
export function inspectableClient(): InspectableClient {
  const client = new QueryClient({
    logger: SILENT,
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  function Providers({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
  return { client, wrapper: Providers }
}

/** What the client caches for the one query the test made. */
export function cachedQueryData(client: QueryClient): unknown {
  const [query] = client.getQueryCache().getAll()
  return query?.state.data
}

/** What the client holds for the one mutation the test ran. */
export function cachedMutationData(client: QueryClient): unknown {
  const [mutation] = client.getMutationCache().getAll()
  return mutation?.state.data
}

/** A fresh copy of a backend answer, as the network would hand it over (never the test's constant). */
export function answer<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}
