'use client'

import { notFound } from 'next/navigation'
import { isEnterprise } from '@/lib/edition-name'
import { PhoneSignInConfirm } from '@/components/enterprise/phone-sign-in-confirm'

/**
 * PRD-277 US-603 — where a phone's sign-in waits for its person's Continue (the API's
 * phone_sign_in.CONFIRM_PAGE). Public, as every page under /sign-in is (lib/public-routes.ts).
 * Only the enterprise edition signs phones in this way; elsewhere the page is not there.
 */
export default function PhoneSignInPage() {
  if (!isEnterprise) {
    notFound()
  }
  return <PhoneSignInConfirm />
}
