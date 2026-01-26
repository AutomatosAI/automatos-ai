import { redirect } from 'next/navigation'

// Legacy route. Canonical sign-in URL is `/sign-in`.
export default function SignInCatchAllPage({
  params,
}: {
  params: { rest?: string[] }
}) {
  const suffix = params?.rest?.length ? `/${params.rest.join('/')}` : ''
  redirect(`/sign-in${suffix}`)
}

