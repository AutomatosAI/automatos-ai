import { redirect } from 'next/navigation'

// Legacy route. Canonical sign-in URL is `/sign-in`.
export default async function SignInCatchAllPage({
  params,
}: {
  params: Promise<{ rest?: string[] }>
}) {
  const { rest } = await params
  const suffix = rest?.length ? `/${rest.join('/')}` : ''
  redirect(`/sign-in${suffix}`)
}
