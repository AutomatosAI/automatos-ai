'use client'

import { useState } from 'react'
import { useClerk, useSignUp } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardFooter } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import Link from 'next/link'
import {
    OAuthSignUpButtons,
    SignUpEmailForm,
    SignUpHeading,
    VerifyEmailForm,
    WaitlistJoined,
    type OAuthSignUpStrategy,
} from './sign-up-parts'

// Clerk's error code for a sign-up refused because the instance is in waitlist mode.
const WAITLIST_RESTRICTED_CODE = 'sign_up_restricted_waitlist'
const WAITLIST_JOIN_FAILED = 'Could not add you to the waitlist. Please try again.'

export function SignUpForm() {
    const { isLoaded, signUp, setActive } = useSignUp()
    const clerk = useClerk()
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [pendingVerification, setPendingVerification] = useState(false)
    const [onWaitlist, setOnWaitlist] = useState(false)
    const [code, setCode] = useState('')
    const [error, setError] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const router = useRouter()

    // Handle OAuth sign up
    const signUpWith = (strategy: OAuthSignUpStrategy) => {
        if (!isLoaded) return

        return signUp.authenticateWithRedirect({
            strategy,
            redirectUrl: '/sso-callback',
            redirectUrlComplete: '/'
        })
    }

    // Clerk billing report (19 Oct 2026) bug 3: waitlist mode refused the sign-up and the form said
    // "on the list" without ever joining it, so the email was lost. Join Clerk's waitlist with the
    // same email, the call Studio's site makes (CTASection: clerk.joinWaitlist).
    const joinWaitlist = async () => {
        try {
            await clerk.joinWaitlist({ emailAddress: email })
            setOnWaitlist(true)
        } catch (err: any) {
            const errorMessage = err.errors?.[0]?.longMessage || err.errors?.[0]?.message || ''
            console.error('Waitlist Error:', errorMessage || err)
            setError(errorMessage || WAITLIST_JOIN_FAILED)
        }
    }

    // Handle Form Submission
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!isLoaded) return

        setIsLoading(true)
        setError(null)

        try {
            await signUp.create({ emailAddress: email, password })

            // Normal flow: send email verification code
            await signUp.prepareEmailAddressVerification({ strategy: 'email_code' })
            setPendingVerification(true)
        } catch (err: any) {
            const errorCode = err.errors?.[0]?.code || ''
            const errorMessage = err.errors?.[0]?.longMessage || err.errors?.[0]?.message || ''

            // Clerk returns this error code when waitlist is active
            if (errorCode === WAITLIST_RESTRICTED_CODE) {
                await joinWaitlist()
                return
            }

            console.error('Sign Up Error:', errorMessage)
            setError(errorMessage || 'Failed to create account. Please try again.')
        } finally {
            setIsLoading(false)
        }
    }

    // Handle Verification
    const handleVerify = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!isLoaded) return

        setIsLoading(true)
        setError(null)

        try {
            const completeSignUp = await signUp.attemptEmailAddressVerification({ code })

            if (completeSignUp.status === 'complete') {
                await setActive({ session: completeSignUp.createdSessionId })
                router.push('/')
            } else {
                console.error('Verification failed', JSON.stringify(completeSignUp, null, 2))
                setError('Verification failed. Use the code sent to your email.')
            }
        } catch (err: any) {
            console.error('Verification Error:', err.errors?.[0]?.longMessage)
            setError(err.errors?.[0]?.longMessage || 'Failed to verify. Please try again.')
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="w-full max-w-md"
        >
            <Card className="overflow-hidden border-border bg-card">
                <CardHeader className="space-y-1 text-center pb-8 pt-8">
                    <SignUpHeading email={email} onWaitlist={onWaitlist} pendingVerification={pendingVerification} />
                </CardHeader>
                <CardContent className="space-y-6">
                    {error && (
                        <Alert variant="destructive" className="bg-destructive/10 border-destructive/20 text-destructive/70">
                            <AlertCircle className="h-4 w-4" />
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    <AnimatePresence mode="wait">
                        {onWaitlist ? (
                            <motion.div
                                key="waitlist-success"
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="text-center py-6 space-y-4"
                            >
                                <WaitlistJoined />
                            </motion.div>
                        ) : !pendingVerification ? (
                            <motion.div
                                key="signup-form"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                className="space-y-6"
                            >
                                <OAuthSignUpButtons onSelect={signUpWith} />
                                <SignUpEmailForm email={email} password={password} isLoading={isLoading}
                                    onEmailChange={setEmail} onPasswordChange={setPassword} onSubmit={handleSubmit} />
                            </motion.div>
                        ) : (
                            <motion.div
                                key="verification-form"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                className="space-y-6"
                            >
                                <VerifyEmailForm code={code} isLoading={isLoading} onCodeChange={setCode} onSubmit={handleVerify} />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </CardContent>
                <CardFooter className="flex justify-center border-t border-border/30 pt-6">
                    <p className="text-sm text-muted-foreground">
                        Already have an invite?{' '}
                        <Link href="/sign-in" className="text-primary hover:text-primary/90 font-medium hover:underline transition-all">
                            Sign in
                        </Link>
                    </p>
                </CardFooter>
            </Card>
        </motion.div >
    )
}
