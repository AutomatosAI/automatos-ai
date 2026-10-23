/**
 * PRD-265 W2 — the card on file and the invoices, in the Billing tab (owner/admin only).
 *
 * - `GET /api/v1/billing/payment-method`: the card the workspace pays with (brand, last four
 *   digits, expiry), or null.
 * - Changing it: `POST …/payment-method/setup` answers a SetupIntent's client secret and the
 *   publishable key; the tab's own card form (Stripe's card field inside it) confirms it with
 *   Stripe.js, then `POST …/payment-method {setup_intent_id}` makes the new card the default
 *   for the customer and the subscription.
 * - `GET /api/v1/billing/invoices`: the last 24 invoices, newest first. Each PDF is served by
 *   our API (`GET …/invoices/{id}/pdf`) and fetched through apiClient, so the browser never
 *   visits Stripe for it.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { isSaaS } from '@/lib/auth-edition'
import { billingErrorMessage } from '@/lib/billing-errors'
import { downloadBlob } from '@/lib/download-text'
import { isClientSecret, isPublishableKey } from '@/lib/stripe'

export const PAYMENT_METHOD_PATH = '/api/v1/billing/payment-method'
export const PAYMENT_METHOD_SETUP_PATH = '/api/v1/billing/payment-method/setup'
export const INVOICES_PATH = '/api/v1/billing/invoices'

export const CARD_FORM_FAILED = 'The card form could not open. Try again in a moment.'
export const CARD_SAVE_FAILED = "The new card couldn't be saved. Try again in a moment."
export const CARD_SAVED_TOAST = 'Card updated.'
export const INVOICE_DOWNLOAD_FAILED = "The invoice couldn't be downloaded. Try again in a moment."

const STALE_TIME_MS = 30_000

export interface CardOnFile {
  brand: string
  last4: string
  exp_month: number
  exp_year: number
}

export interface Invoice {
  id: string
  number: string | null
  status: string
  total_cents: number
  currency: string
  created: string
}

export interface CardSetup {
  client_secret: string
  publishable_key: string
}

export const billingPaymentQueryKeys = {
  method: ['billing', 'payment-method'] as const,
  invoices: ['billing', 'invoices'] as const,
}

export function invoicePdfPath(invoiceId: string): string {
  return `${INVOICES_PATH}/${encodeURIComponent(invoiceId)}/pdf`
}

/** The card on file (null for none). */
export function usePaymentMethod(enabled: boolean) {
  return useQuery<CardOnFile | null, ApiHttpError>({
    queryKey: billingPaymentQueryKeys.method,
    queryFn: async () => (await apiClient.request<{ payment_method: CardOnFile | null }>(PAYMENT_METHOD_PATH))?.payment_method ?? null,
    enabled: isSaaS && enabled,
    staleTime: STALE_TIME_MS,
  })
}

/** The workspace's invoices, newest first. */
export function useInvoices(enabled: boolean) {
  return useQuery<Invoice[], ApiHttpError>({
    queryKey: billingPaymentQueryKeys.invoices,
    queryFn: async () => (await apiClient.request<{ items: Invoice[] }>(INVOICES_PATH))?.items ?? [],
    enabled: isSaaS && enabled,
    staleTime: STALE_TIME_MS,
  })
}

async function requestCardSetup(): Promise<CardSetup> {
  const answer = await apiClient.post<CardSetup>(PAYMENT_METHOD_SETUP_PATH)
  const secret = answer?.client_secret
  const key = answer?.publishable_key
  if (!isClientSecret(secret, 'seti_') || !isPublishableKey(key)) throw new Error(CARD_FORM_FAILED)
  return { client_secret: secret, publishable_key: key }
}

/** Start a card change: the SetupIntent the card form confirms. */
export function useCardSetup() {
  return useMutation<CardSetup, ApiHttpError, void>({
    mutationFn: requestCardSetup,
    onError: (error) => {
      toast.error(billingErrorMessage(error, CARD_FORM_FAILED))
    },
  })
}

/** Make the card a confirmed SetupIntent collected the workspace's default; errors are the form's to show. */
export function useSavePaymentMethod() {
  const queryClient = useQueryClient()
  return useMutation<CardOnFile | null, ApiHttpError, string>({
    mutationFn: async (setupIntentId) =>
      (await apiClient.post<{ payment_method: CardOnFile | null }>(PAYMENT_METHOD_PATH, { setup_intent_id: setupIntentId }))
        ?.payment_method ?? null,
    onSuccess: (card) => {
      queryClient.setQueryData(billingPaymentQueryKeys.method, card)
      toast.success(CARD_SAVED_TOAST)
    },
  })
}

/** Save an invoice's PDF, fetched through apiClient. */
export async function downloadInvoicePdf(invoice: Pick<Invoice, 'id' | 'number'>): Promise<void> {
  const blob = await apiClient.requestBlob(invoicePdfPath(invoice.id))
  downloadBlob(`${invoice.number || invoice.id}.pdf`, blob)
}
