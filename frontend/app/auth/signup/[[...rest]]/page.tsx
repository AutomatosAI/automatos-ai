import { redirect } from 'next/navigation'

// Legacy route. Canonical sign-up URL is `/sign-up`.
export default function SignUpCatchAllPage({
  params,
}: {
  params: { rest?: string[] }
}) {
  const suffix = params?.rest?.length ? `/${params.rest.join('/')}` : ''
  redirect(`/sign-up${suffix}`)
}

