'use client'

import { useState } from 'react'
import { useSignUp } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { Eye, EyeOff, Lock, Mail, ArrowRight, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import Link from 'next/link'
import Image from 'next/image'

export function SignUpForm() {
    const { isLoaded, signUp, setActive } = useSignUp()
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [showPassword, setShowPassword] = useState(false)
    const [pendingVerification, setPendingVerification] = useState(false)
    const [onWaitlist, setOnWaitlist] = useState(false)
    const [code, setCode] = useState('')
    const [error, setError] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const router = useRouter()

    // Handle OAuth sign up
    const signUpWith = (strategy: 'oauth_google' | 'oauth_github') => {
        if (!isLoaded) return

        return signUp.authenticateWithRedirect({
            strategy,
            redirectUrl: '/sso-callback',
            redirectUrlComplete: '/'
        })
    }

    // Handle Form Submission
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!isLoaded) return

        setIsLoading(true)
        setError(null)

        try {
            const result = await signUp.create({
                emailAddress: email,
                password,
            })

            // Check if user was placed on waitlist
            if (result.status === 'missing_requirements' &&
                result.unverifiedFields?.includes('email_address') === false) {
                setOnWaitlist(true)
                return
            }

            // Normal flow: send email verification code
            await signUp.prepareEmailAddressVerification({ strategy: 'email_code' })
            setPendingVerification(true)
        } catch (err: any) {
            const errorMessage = err.errors?.[0]?.longMessage || err.errors?.[0]?.message || ''

            // Clerk returns specific errors when waitlist is active
            if (errorMessage.toLowerCase().includes('waitlist')) {
                setOnWaitlist(true)
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
            const completeSignUp = await signUp.attemptEmailAddressVerification({
                code,
            })

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
            <Card className="glass-card overflow-hidden border-border/50 shadow-2xl">
                <CardHeader className="space-y-1 text-center pb-8 pt-8">
                    <motion.div
                        initial={{ y: -20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.1 }}
                        className="flex justify-center mb-4"
                    >
                        {/* Brand Logo */}
                        <Image
                            src="/brand/automatos-mark.png"
                            alt="Automatos AI"
                            width={48}
                            height={48}
                            className="rounded-xl shadow-lg shadow-orange-500/20"
                        />
                    </motion.div>
                    <CardTitle className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                        {onWaitlist ? "You're on the list!" : pendingVerification ? 'Verify Email' : 'Join the Waitlist'}
                    </CardTitle>
                    <CardDescription className="text-slate-400">
                        {onWaitlist
                            ? "We'll notify you at " + email + " when your spot is ready"
                            : pendingVerification
                                ? 'We sent a verification code to ' + email
                                : 'Sign up to join the Automatos waitlist'}
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    {error && (
                        <Alert variant="destructive" className="bg-red-500/10 border-red-500/20 text-red-200">
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
                                <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mx-auto">
                                    <CheckCircle2 className="w-8 h-8 text-primary" />
                                </div>
                                <p className="text-sm text-slate-400">
                                    We're onboarding users in waves. You'll receive an invite email when it's your turn.
                                </p>
                                <Link href="/sign-in">
                                    <Button variant="outline" className="mt-2 border-border/40 hover:bg-secondary/40">
                                        Already have an invite? Sign in
                                    </Button>
                                </Link>
                            </motion.div>
                        ) : !pendingVerification ? (
                            <motion.div
                                key="signup-form"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                className="space-y-6"
                            >
                                <div className="grid grid-cols-2 gap-4">
                                    <Button
                                        variant="outline"
                                        className="bg-secondary/20 border-border/40 hover:bg-secondary/40 hover:text-white transition-all duration-200"
                                        onClick={() => signUpWith('oauth_github')}
                                        type="button"
                                    >
                                        <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                                            <path
                                                d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
                                                fill="currentColor"
                                            />
                                        </svg>
                                        GitHub
                                    </Button>
                                    <Button
                                        variant="outline"
                                        className="bg-secondary/20 border-border/40 hover:bg-secondary/40 hover:text-white transition-all duration-200"
                                        onClick={() => signUpWith('oauth_google')}
                                        type="button"
                                    >
                                        <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                                            <path
                                                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                                                fill="#4285F4"
                                            />
                                            <path
                                                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                                                fill="#34A853"
                                            />
                                            <path
                                                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                                                fill="#FBBC05"
                                            />
                                            <path
                                                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                                                fill="#EA4335"
                                            />
                                        </svg>
                                        Google
                                    </Button>
                                </div>

                                <div className="relative">
                                    <div className="absolute inset-0 flex items-center">
                                        <span className="w-full border-t border-border/30" />
                                    </div>
                                    <div className="relative flex justify-center text-xs uppercase">
                                        <span className="bg-background px-2 text-muted-foreground bg-black/50 backdrop-blur-sm rounded">
                                            Or register with
                                        </span>
                                    </div>
                                </div>

                                <form onSubmit={handleSubmit} className="space-y-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="email" className="text-slate-300">Email</Label>
                                        <div className="relative group">
                                            <Mail className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                                            <Input
                                                id="email"
                                                type="email"
                                                placeholder="name@example.com"
                                                value={email}
                                                onChange={(e) => setEmail(e.target.value)}
                                                className="pl-9 bg-secondary/20 border-border/40 focus:border-primary/50 focus:bg-secondary/30 transition-all"
                                                required
                                            />
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="password" className="text-slate-300">Password</Label>
                                        <div className="relative group">
                                            <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                                            <Input
                                                id="password"
                                                type={showPassword ? 'text' : 'password'}
                                                value={password}
                                                onChange={(e) => setPassword(e.target.value)}
                                                className="pl-9 pr-9 bg-secondary/20 border-border/40 focus:border-primary/50 focus:bg-secondary/30 transition-all"
                                                required
                                                minLength={8}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowPassword(!showPassword)}
                                                className="absolute right-3 top-2.5 text-muted-foreground hover:text-white transition-colors"
                                            >
                                                {showPassword ? (
                                                    <EyeOff className="h-4 w-4" />
                                                ) : (
                                                    <Eye className="h-4 w-4" />
                                                )}
                                            </button>
                                        </div>
                                    </div>

                                    <Button
                                        type="submit"
                                        className="w-full gradient-accent font-medium shadow-lg shadow-orange-500/20 hover:shadow-orange-500/40 transition-all duration-300"
                                        disabled={isLoading}
                                    >
                                        {isLoading ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Creating account...
                                            </>
                                        ) : (
                                            <>
                                                Join Waitlist
                                                <ArrowRight className="ml-2 h-4 w-4" />
                                            </>
                                        )}
                                    </Button>
                                </form>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="verification-form"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                className="space-y-6"
                            >
                                <form onSubmit={handleVerify} className="space-y-6">
                                    <div className="flex justify-center py-4">
                                        <div className="space-y-2 text-center">
                                            <Label htmlFor="code" className="sr-only">Verification Code</Label>
                                            <div className="relative">
                                                <Input
                                                    id="code"
                                                    value={code}
                                                    onChange={(e) => setCode(e.target.value)}
                                                    maxLength={6}
                                                    className="text-center text-2xl tracking-[1em] font-mono h-14 bg-secondary/20 border-border/40 rounded-xl focus:ring-2 ring-primary/50 transition-all uppercase placeholder:tracking-normal"
                                                    placeholder="000000"
                                                    autoComplete="one-time-code"
                                                />
                                            </div>
                                            <p className="text-xs text-muted-foreground pt-2">
                                                Enter the 6-digit code sent to your email
                                            </p>
                                        </div>
                                    </div>

                                    <Button
                                        type="submit"
                                        className="w-full gradient-accent font-medium shadow-lg shadow-orange-500/20 hover:shadow-orange-500/40 transition-all duration-300"
                                        disabled={isLoading || code.length < 6}
                                    >
                                        {isLoading ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Verifying...
                                            </>
                                        ) : (
                                            <>
                                                Verify Email
                                                <CheckCircle2 className="ml-2 h-4 w-4" />
                                            </>
                                        )}
                                    </Button>
                                </form>
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
