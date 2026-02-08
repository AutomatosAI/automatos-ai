import { redirect } from 'next/navigation'

// Legacy route. Canonical sign-up URL is `/sign-up`.
export default async function SignUpCatchAllPage({
  params,
}: {
  params: Promise<{ rest?: string[] }>
}) {
  const { rest } = await params
  const suffix = rest?.length ? `/${rest.join('/')}` : ''
  redirect(`/sign-up${suffix}`)
}
