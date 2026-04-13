'use client'

import { useState } from 'react'
import { useSignIn } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { ArrowRight, Loader2, AlertCircle, Lock, KeyRound } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import Image from 'next/image'

export default function ResetPasswordPage() {
    const { isLoaded, signIn, setActive } = useSignIn()
    const [code, setCode] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [error, setError] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const router = useRouter()

    const handleReset = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!isLoaded) return

        setIsLoading(true)
        setError(null)

        try {
            const result = await signIn.attemptFirstFactor({
                strategy: 'reset_password_email_code',
                code,
                password: newPassword,
            })

            if (result.status === 'complete') {
                await setActive({ session: result.createdSessionId })
                router.push('/')
            } else {
                setError('Could not complete password reset. Please try again.')
            }
        } catch (err: any) {
            const msg = err.errors?.[0]?.longMessage || err.errors?.[0]?.message
            setError(msg || 'Password reset failed. Please check your code and try again.')
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-black via-slate-950 to-black">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
                className="w-full max-w-md"
            >
                <Card className="glass-card overflow-hidden border-border/50 shadow-2xl">
                    <CardHeader className="space-y-1 text-center pb-6 pt-8">
                        <motion.div
                            initial={{ y: -20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ delay: 0.1 }}
                            className="flex justify-center mb-4"
                        >
                            <Image
                                src="/brand/automatos-mark.png"
                                alt="Automatos AI"
                                width={48}
                                height={48}
                                className="rounded-xl shadow-lg shadow-orange-500/20"
                            />
                        </motion.div>
                        <CardTitle className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                            Reset Password
                        </CardTitle>
                        <CardDescription className="text-slate-400">
                            Check your email for a verification code
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        {error && (
                            <Alert variant="destructive" className="bg-red-500/10 border-red-500/20 text-red-200">
                                <AlertCircle className="h-4 w-4" />
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        )}

                        <form onSubmit={handleReset} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="code" className="text-slate-300">Verification Code</Label>
                                <div className="relative group">
                                    <KeyRound className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                                    <Input
                                        id="code"
                                        type="text"
                                        placeholder="Enter code from email"
                                        value={code}
                                        onChange={(e) => setCode(e.target.value)}
                                        className="pl-9 bg-secondary/20 border-border/40 focus:border-primary/50 focus:bg-secondary/30 transition-all"
                                        required
                                    />
                                </div>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="new-password" className="text-slate-300">New Password</Label>
                                <div className="relative group">
                                    <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                                    <Input
                                        id="new-password"
                                        type="password"
                                        placeholder="Enter new password"
                                        value={newPassword}
                                        onChange={(e) => setNewPassword(e.target.value)}
                                        className="pl-9 bg-secondary/20 border-border/40 focus:border-primary/50 focus:bg-secondary/30 transition-all"
                                        required
                                        minLength={8}
                                    />
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
                                        Resetting...
                                    </>
                                ) : (
                                    <>
                                        Reset Password
                                        <ArrowRight className="ml-2 h-4 w-4" />
                                    </>
                                )}
                            </Button>
                        </form>
                    </CardContent>
                </Card>
            </motion.div>
        </div>
    )
}
