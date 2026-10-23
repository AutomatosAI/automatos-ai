'use client'

/**
 * PRD-265 W2 — the buyer's VAT number, in our own fields (Stripe's Tax ID Element is a beta).
 *
 * Shown only for a billing address in the UK (a `gb_vat` number) or an EU country (`eu_vat`).
 * "Add VAT number" hands it and the business name to the session
 * (`checkout.updateTaxIdInfo`), which works the tax out again (a valid EU business number is
 * reverse-charged); Stripe's answer to a number it cannot use is shown as it is. Moving the
 * address to a country the number does not belong to takes the number off the session.
 */
import { useEffect, useState } from 'react'
import type { StripeCheckoutElementsValue } from '@stripe/react-stripe-js/checkout'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { stripeErrorMessage } from '@/lib/billing-errors'

export const VAT_FAILED = "That VAT number couldn't be used. Check it and try again."
export const VAT_ADDED = 'VAT number added.'
export const VAT_REMOVED = 'VAT number removed.'

export type VatIdType = 'gb_vat' | 'eu_vat'

/** The EU's member states, by ISO 3166-1 alpha-2 code. */
const EU_COUNTRIES: ReadonlySet<string> = new Set([
  'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE',
  'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
])
const UK = 'GB'

/** The VAT number kind for a billing country; null where the field does not show. */
export function vatIdTypeFor(country: string | null | undefined): VatIdType | null {
  const code = (country ?? '').toUpperCase()
  if (code === UK) return 'gb_vat'
  return EU_COUNTRIES.has(code) ? 'eu_vat' : null
}

type TaxIdSession = Pick<StripeCheckoutElementsValue, 'taxIdInfo' | 'updateTaxIdInfo'>

interface VatFieldProps {
  readonly checkout: TaxIdSession
  readonly country: string | null
}

/** A number already on the session that no longer fits the billing country comes off. */
function useDropMismatchedNumber(checkout: TaxIdSession, type: VatIdType | null): void {
  const onSession = checkout.taxIdInfo?.taxId.type ?? null
  const { updateTaxIdInfo } = checkout
  useEffect(() => {
    if (onSession !== null && onSession !== type) void updateTaxIdInfo(null)
  }, [onSession, type, updateTaxIdInfo])
}

export function VatField({ checkout, country }: VatFieldProps) {
  const type = vatIdTypeFor(country)
  const [businessName, setBusinessName] = useState(checkout.taxIdInfo?.businessName ?? '')
  const [number, setNumber] = useState(checkout.taxIdInfo?.taxId.value ?? '')
  const [note, setNote] = useState<{ failed: boolean; text: string } | null>(null)
  const [saving, setSaving] = useState(false)
  useDropMismatchedNumber(checkout, type)
  if (!type) return null

  const idType: VatIdType = type
  const value = number.trim()
  const name = businessName.trim()
  const apply = async () => {
    setSaving(true)
    try {
      const result = await checkout.updateTaxIdInfo(value ? { taxId: { type: idType, value }, businessName: name } : null)
      const failed = result.type !== 'success'
      const text = result.type === 'success' ? (value ? VAT_ADDED : VAT_REMOVED) : stripeErrorMessage(result.error, VAT_FAILED)
      setNote({ failed, text })
    } catch (failure) {
      setNote({ failed: true, text: stripeErrorMessage(failure as { message?: unknown }, VAT_FAILED) })
    } finally {
      setSaving(false)
    }
  }

  return (
    <fieldset data-testid="billing-vat" className="space-y-3 rounded-xl border border-border p-3">
      <legend className="px-1 text-xs uppercase tracking-wide text-muted-foreground">VAT number (optional)</legend>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1">
          <Label htmlFor="billing-vat-business">Business name</Label>
          <Input id="billing-vat-business" value={businessName} onChange={(event) => setBusinessName(event.target.value)} />
        </div>
        <div className="space-y-1">
          <Label htmlFor="billing-vat-number">VAT number</Label>
          <Input id="billing-vat-number" value={number} onChange={(event) => setNumber(event.target.value)} />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {(value !== '' || checkout.taxIdInfo) && (
          <Button type="button" size="sm" variant="outline" disabled={saving || (value !== '' && name === '')} onClick={() => void apply()}>
            {value ? 'Add VAT number' : 'Remove VAT number'}
          </Button>
        )}
        {note && (
          <p role={note.failed ? 'alert' : 'status'} className={note.failed ? 'text-sm text-destructive' : 'text-sm text-muted-foreground'}>
            {note.text}
          </p>
        )}
      </div>
    </fieldset>
  )
}
