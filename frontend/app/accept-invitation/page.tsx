'use client'

import { useEffect, useState, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuth, useUser, SignIn, SignUp } from '@clerk/nextjs'
import { apiClient } from '@/lib/api-client'
import { CheckCircle2, AlertCircle, Loader2, Mail } from 'lucide-react'
import Link from 'next/link'

interface InvitationInfo {
    email: string
    role: string
    workspace_name: string
    inviter_name: string | null
    expires_at: string
    expired: boolean
    accepted: boolean
}

interface AcceptResponse {
    workspace_id: string
    workspace_name: string
    role: string
    already_member: boolean
}

function AcceptInvitationInner() {
    const router = useRouter()
    const searchParams = useSearchParams()
    const token = searchParams.get('token')

    const { isLoaded: authLoaded, isSignedIn } = useAuth()
    const { user } = useUser()

    const [info, setInfo] = useState<InvitationInfo | null>(null)
    const [infoError, setInfoError] = useState<string | null>(null)
    const [accepting, setAccepting] = useState(false)
    const [accepted, setAccepted] = useState<AcceptResponse | null>(null)
    const [acceptError, setAcceptError] = useState<string | null>(null)

    // Look up the invitation metadata (public, no auth) so we can show
    // workspace name + inviter even before sign-in.
    useEffect(() => {
        if (!token) {
            setInfoError('No invitation token provided in the link.')
            return
        }
        const baseUrl = process.env.NEXT_PUBLIC_API_URL || ''
        fetch(`${baseUrl}/api/team/invitation-info?token=${encodeURIComponent(token)}`)
            .then(async (r) => {
                if (!r.ok) {
                    const body = await r.json().catch(() => ({}))
                    throw new Error(body.detail || `HTTP ${r.status}`)
                }
                return r.json()
            })
            .then(setInfo)
            .catch((e: Error) => setInfoError(e.message || 'Could not load invitation'))
    }, [token])

    // Once Clerk is loaded and the user is signed in, exchange the token.
    // CRITICAL: acceptError MUST gate this effect — without it, a server
    // failure causes an infinite POST loop (setAccepting(false) re-fires
    // the effect, which retries, which fails, which re-fires…).
    useEffect(() => {
        if (!authLoaded || !isSignedIn || !token) return
        if (accepted || accepting || acceptError) return
        if (!info || info.expired || info.accepted) return

        setAccepting(true)
        setAcceptError(null)

        apiClient
            .request<AcceptResponse>('/api/team/accept-invitation', {
                method: 'POST',
                body: { token },
            })
            .then((res) => {
                setAccepted(res)
                if (typeof window !== 'undefined') {
                    localStorage.setItem('last_active_workspace', res.workspace_id)
                }
                setTimeout(() => router.push('/dashboard'), 1500)
            })
            .catch((e: Error) => {
                setAcceptError(e.message || 'Failed to accept invitation')
            })
            .finally(() => setAccepting(false))
    }, [authLoaded, isSignedIn, token, info, accepted, accepting, acceptError, router])

    if (!token) {
        return <ErrorState title="Invalid link" detail="This invitation link is missing a token." />
    }

    if (infoError) {
        return <ErrorState title="Invitation not found" detail={infoError} />
    }

    if (!info || !authLoaded) {
        return (
            <Shell>
                <div className="flex flex-col items-center text-center gap-4">
                    <Loader2 className="w-8 h-8 animate-spin text-orange-500" />
                    <p className="text-muted-foreground">Loading invitation…</p>
                </div>
            </Shell>
        )
    }

    if (info.expired) {
        return <ErrorState title="Invitation expired" detail="This invitation is past its expiration date. Ask the workspace admin to send a new one." />
    }

    if (info.accepted && !accepted) {
        return <ErrorState title="Already accepted" detail="This invitation has already been used. Sign in to access the workspace." cta={{ href: '/sign-in', label: 'Go to sign-in' }} />
    }

    if (accepted) {
        return (
            <Shell>
                <div className="flex flex-col items-center text-center gap-4">
                    <CheckCircle2 className="w-12 h-12 text-green-500" />
                    <h1 className="text-2xl font-semibold">You're in!</h1>
                    <p className="text-muted-foreground">
                        Joined <span className="text-foreground font-medium">{accepted.workspace_name}</span> as{' '}
                        <span className="text-foreground font-medium">{accepted.role}</span>.
                    </p>
                    <p className="text-sm text-muted-foreground">Redirecting to your dashboard…</p>
                </div>
            </Shell>
        )
    }

    if (!isSignedIn) {
        // Pre-fill email and offer sign-up; Clerk will route the user back here.
        const redirectBack = `/accept-invitation?token=${encodeURIComponent(token)}`
        return (
            <Shell wide>
                <div className="space-y-6">
                    <div className="text-center space-y-2">
                        <Mail className="w-8 h-8 text-orange-500 mx-auto" />
                        <h1 className="text-2xl font-semibold">
                            Join {info.workspace_name}
                        </h1>
                        <p className="text-muted-foreground text-sm">
                            {info.inviter_name ? `${info.inviter_name} invited you` : "You've been invited"} to collaborate as{' '}
                            <span className="text-foreground font-medium">{info.role}</span>.
                        </p>
                        <p className="text-xs text-muted-foreground/70">
                            Sign up with <span className="text-foreground">{info.email}</span> to accept.
                        </p>
                    </div>

                    <SignUp
                        appearance={{
                            elements: {
                                rootBox: 'mx-auto',
                                card: 'bg-zinc-900/80 border border-zinc-800 shadow-2xl',
                            },
                        }}
                        initialValues={{ emailAddress: info.email }}
                        redirectUrl={redirectBack}
                        signInUrl={`/sign-in?redirect_url=${encodeURIComponent(redirectBack)}`}
                    />
                </div>
            </Shell>
        )
    }

    // Signed in but accept either still in-flight or has errored.
    if (acceptError) {
        // Email mismatch is the most common cause; surface a hint.
        const isMismatch = user?.primaryEmailAddress?.emailAddress?.toLowerCase() !== info.email.toLowerCase()
        return (
            <Shell>
                <div className="flex flex-col items-center text-center gap-4">
                    <AlertCircle className="w-12 h-12 text-red-500" />
                    <h1 className="text-2xl font-semibold">Could not accept invitation</h1>
                    <p className="text-muted-foreground text-sm">{acceptError}</p>
                    {isMismatch && user?.primaryEmailAddress?.emailAddress && (
                        <p className="text-xs text-amber-400">
                            You're signed in as <span className="font-medium">{user.primaryEmailAddress.emailAddress}</span>,
                            but the invite was sent to <span className="font-medium">{info.email}</span>. Sign out and use the invited address.
                        </p>
                    )}
                    <Link
                        href="/dashboard"
                        className="text-sm text-orange-500 hover:text-orange-400"
                    >
                        Go to dashboard
                    </Link>
                </div>
            </Shell>
        )
    }

    return (
        <Shell>
            <div className="flex flex-col items-center text-center gap-4">
                <Loader2 className="w-8 h-8 animate-spin text-orange-500" />
                <p className="text-muted-foreground">Joining {info.workspace_name}…</p>
            </div>
        </Shell>
    )
}

function Shell({ children, wide = false }: { children: React.ReactNode; wide?: boolean }) {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black via-zinc-900 to-black px-6 py-12 relative overflow-hidden">
            <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full bg-primary/10 blur-[120px] pointer-events-none" />
            <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] rounded-full bg-orange-600/10 blur-[120px] pointer-events-none" />
            <div className={`relative z-10 w-full ${wide ? 'max-w-md' : 'max-w-sm'} bg-[#0A0A0A]/60 border border-white/10 rounded-xl p-8 backdrop-blur-md`}>
                {children}
            </div>
        </div>
    )
}

function ErrorState({ title, detail, cta }: { title: string; detail: string; cta?: { href: string; label: string } }) {
    return (
        <Shell>
            <div className="flex flex-col items-center text-center gap-4">
                <AlertCircle className="w-12 h-12 text-red-500" />
                <h1 className="text-2xl font-semibold">{title}</h1>
                <p className="text-muted-foreground text-sm">{detail}</p>
                <Link
                    href={cta?.href || '/sign-in'}
                    className="text-sm text-orange-500 hover:text-orange-400"
                >
                    {cta?.label || 'Go to sign-in'}
                </Link>
            </div>
        </Shell>
    )
}

export default function AcceptInvitationPage() {
    return (
        <Suspense fallback={<Shell><Loader2 className="w-8 h-8 animate-spin text-orange-500 mx-auto" /></Shell>}>
            <AcceptInvitationInner />
        </Suspense>
    )
}
