'use client'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
    AlertCircle,
    ArrowLeft,
    Briefcase,
    Check,
    CheckCircle,
    Copy,
    Loader2,
    Mail,
    Save,
    User,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PageHeader } from '@/components/shared'
import { useWorkspace } from '@/hooks/use-workspace'
import { apiClient, type ProfileResponse, type ProfileUpdateRequest } from '@/lib/api-client'
import { initialsFor } from '@/components/auth/user-profile-button'

/**
 * PRD-233 S6 — the local operator's profile.
 *
 * The local edition has exactly one operator and no login, so there is no
 * identity provider to edit a profile in. This form edits the operator's
 * `users` row through GET/PUT /api/profile: display name (the name Auto greets
 * you by), username and avatar URL. Email is read-only — it is the lookup key
 * the anonymous session resolves the row by (`LOCAL_OPERATOR_EMAIL` in .env),
 * and the backend says so in `email_note`.
 */

const STATUS_RESET_MS = 3000
const COPY_RESET_MS = 2000
const INPUT_CLASS =
    'w-full px-4 py-2 bg-secondary/50 border border-border rounded-lg text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none'
const READONLY_CLASS =
    'w-full px-4 py-2 bg-secondary/30 border border-border rounded-lg text-muted-foreground cursor-not-allowed'

interface FormFields {
    name: string
    username: string
    avatar_url: string
}

const EMPTY_FORM: FormFields = { name: '', username: '', avatar_url: '' }

function formFrom(profile: ProfileResponse): FormFields {
    return {
        name: profile.name ?? '',
        username: profile.username ?? '',
        avatar_url: profile.avatar_url ?? '',
    }
}

/** Only the fields that differ from the loaded profile — PUT is partial. */
export function changedFields(profile: ProfileResponse, form: FormFields): ProfileUpdateRequest {
    const current = formFrom(profile)
    return (Object.keys(form) as Array<keyof FormFields>).reduce<ProfileUpdateRequest>(
        (acc, key) => (form[key] === current[key] ? acc : { ...acc, [key]: form[key] }),
        {},
    )
}

/** api-client throws `Error(detail)`; a FastAPI 422 detail arrives JSON-stringified. */
export function readableError(error: unknown): string {
    const message = error instanceof Error ? error.message : String(error)
    if (!message.startsWith('[')) return message
    try {
        const items = JSON.parse(message) as Array<{ loc?: unknown[]; msg?: string }>
        return items
            .map((item) => {
                const field = item.loc && item.loc.length > 0 ? String(item.loc[item.loc.length - 1]) : 'field'
                return `${field}: ${item.msg ?? 'invalid'}`
            })
            .join('; ')
    } catch {
        return message
    }
}

export function LocalProfileForm() {
    const router = useRouter()
    const { workspaceId, loading: workspaceLoading } = useWorkspace()
    const [profile, setProfile] = useState<ProfileResponse | null>(null)
    const [form, setForm] = useState<FormFields>(EMPTY_FORM)
    const [loadError, setLoadError] = useState<string | null>(null)
    const [isSaving, setIsSaving] = useState(false)
    const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
    const [saveError, setSaveError] = useState<string | null>(null)
    const [copiedWorkspaceId, setCopiedWorkspaceId] = useState(false)

    useEffect(() => {
        let cancelled = false
        apiClient
            .getProfile()
            .then((data) => {
                if (cancelled) return
                setProfile(data)
                setForm(formFrom(data))
            })
            .catch((error: unknown) => {
                if (!cancelled) setLoadError(readableError(error))
            })
        return () => {
            cancelled = true
        }
    }, [])

    const flash = useCallback((status: 'success' | 'error', message: string | null = null) => {
        setSaveStatus(status)
        setSaveError(message)
        setTimeout(() => {
            setSaveStatus('idle')
            setSaveError(null)
        }, STATUS_RESET_MS)
    }, [])

    const handleSave = async () => {
        if (!profile) return
        const updates = changedFields(profile, form)
        if (Object.keys(updates).length === 0) {
            flash('success')
            return
        }
        setIsSaving(true)
        try {
            const saved = await apiClient.updateProfile(updates)
            setProfile(saved)
            setForm(formFrom(saved))
            flash('success')
        } catch (error: unknown) {
            flash('error', readableError(error))
        } finally {
            setIsSaving(false)
        }
    }

    const handleCopyWorkspaceId = async () => {
        if (!workspaceId) return
        try {
            await navigator.clipboard.writeText(workspaceId)
            setCopiedWorkspaceId(true)
            setTimeout(() => setCopiedWorkspaceId(false), COPY_RESET_MS)
        } catch {
            setCopiedWorkspaceId(false)
        }
    }

    if (loadError) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <p className="text-destructive">Could not load your profile: {loadError}</p>
            </div>
        )
    }

    if (!profile) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary" />
            </div>
        )
    }

    const dirty = Object.keys(changedFields(profile, form)).length > 0
    const displayName = form.name.trim() || profile.email || 'Operator'

    return (
        <div className="min-h-screen py-8 px-4">
            <div className="max-w-4xl mx-auto">
                <div className="mb-8">
                    <PageHeader
                        title="Profile"
                        titleAccent="Settings"
                        eyebrow="Local edition · operator"
                        lede="Who Auto is talking to on this instance. There is no login locally — this is a profile, not an account."
                        actions={
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => router.back()}
                                className="text-muted-foreground hover:text-foreground hover:bg-secondary"
                            >
                                <ArrowLeft className="w-4 h-4 mr-1" />
                                Back
                            </Button>
                        }
                    />
                </div>

                {saveStatus !== 'idle' && (
                    <div
                        role="status"
                        className={`mb-6 p-4 rounded-lg border flex items-center space-x-3 ${saveStatus === 'success'
                            ? 'bg-success/10 border-success/30 text-success'
                            : 'bg-destructive/10 border-destructive/30 text-destructive'
                            }`}
                    >
                        {saveStatus === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                        <span>{saveStatus === 'success' ? 'Profile updated.' : saveError || 'Failed to update profile.'}</span>
                    </div>
                )}

                <form
                    className="glass-card rounded-2xl border border-border p-8 mb-6"
                    onSubmit={(event) => {
                        event.preventDefault()
                        void handleSave()
                    }}
                >
                    <div className="flex items-center space-x-6 mb-8 pb-8 border-b border-border">
                        {form.avatar_url ? (
                            <img
                                src={form.avatar_url}
                                alt={displayName}
                                className="w-24 h-24 rounded-full border-4 border-primary/30 object-cover"
                            />
                        ) : (
                            <div
                                aria-label="Avatar initials"
                                className="w-24 h-24 rounded-full bg-primary flex items-center justify-center text-2xl font-semibold text-primary-foreground"
                            >
                                {initialsFor(form.name, profile.email)}
                            </div>
                        )}
                        <div>
                            <h2 className="text-2xl font-bold text-foreground mb-1">{displayName}</h2>
                            <p className="text-muted-foreground">{profile.email}</p>
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                                <User className="w-5 h-5 mr-2 text-primary" />
                                Personal Information
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="md:col-span-2">
                                    <label htmlFor="profile-name" className="block text-sm font-medium text-foreground mb-2">
                                        Display name
                                    </label>
                                    <input
                                        id="profile-name"
                                        type="text"
                                        value={form.name}
                                        maxLength={255}
                                        onChange={(event) => setForm({ ...form, name: event.target.value })}
                                        className={INPUT_CLASS}
                                        placeholder="The name Auto greets you by"
                                    />
                                    <p className="mt-2 text-xs text-muted-foreground">
                                        Auto greets you by this name. Leave it blank and Auto stays generic.
                                    </p>
                                </div>
                                <div>
                                    <label htmlFor="profile-username" className="block text-sm font-medium text-foreground mb-2">
                                        Username
                                    </label>
                                    <input
                                        id="profile-username"
                                        type="text"
                                        value={form.username}
                                        maxLength={255}
                                        onChange={(event) => setForm({ ...form, username: event.target.value })}
                                        className={INPUT_CLASS}
                                        placeholder="letters, digits, . _ -"
                                    />
                                </div>
                                <div>
                                    <label htmlFor="profile-avatar" className="block text-sm font-medium text-foreground mb-2">
                                        Avatar URL
                                    </label>
                                    <input
                                        id="profile-avatar"
                                        type="url"
                                        value={form.avatar_url}
                                        maxLength={500}
                                        onChange={(event) => setForm({ ...form, avatar_url: event.target.value })}
                                        className={INPUT_CLASS}
                                        placeholder="https://…"
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="pt-6 border-t border-border">
                            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                                <Mail className="w-5 h-5 mr-2 text-primary" />
                                Email
                            </h3>
                            <label htmlFor="profile-email" className="block text-sm font-medium text-foreground mb-2">
                                Email
                            </label>
                            <input
                                id="profile-email"
                                type="email"
                                value={profile.email ?? ''}
                                readOnly
                                aria-readonly="true"
                                className={READONLY_CLASS}
                            />
                            <p className="mt-2 text-xs text-muted-foreground">{profile.email_note}</p>
                        </div>

                        <div className="flex justify-end">
                            <Button
                                type="submit"
                                disabled={isSaving || !dirty}
                                className="bg-primary hover:bg-primary/90 text-primary-foreground disabled:opacity-50"
                            >
                                {isSaving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
                                {isSaving ? 'Saving…' : 'Save'}
                            </Button>
                        </div>
                    </div>
                </form>

                <div className="glass-card rounded-2xl border border-border p-8">
                    <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                        <Briefcase className="w-5 h-5 mr-2 text-primary" />
                        Workspace
                    </h3>
                    <label className="block text-sm font-medium text-foreground mb-2">Workspace ID</label>
                    <div className="flex items-center space-x-2">
                        <div className="flex-1 px-4 py-2 bg-secondary/30 border border-border rounded-lg">
                            {workspaceLoading ? (
                                <span className="text-muted-foreground font-mono text-sm">Loading…</span>
                            ) : workspaceId ? (
                                <span className="text-foreground font-mono text-sm break-all">{workspaceId}</span>
                            ) : (
                                <span className="text-muted-foreground font-mono text-sm">No workspace</span>
                            )}
                        </div>
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleCopyWorkspaceId}
                            disabled={!workspaceId}
                            className="border-border text-foreground hover:bg-secondary hover:text-foreground disabled:opacity-50"
                        >
                            {copiedWorkspaceId ? (
                                <>
                                    <Check className="w-4 h-4 mr-2 text-success" />
                                    Copied
                                </>
                            ) : (
                                <>
                                    <Copy className="w-4 h-4 mr-2" />
                                    Copy
                                </>
                            )}
                        </Button>
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">
                        Include this ID when reporting an issue — it identifies this instance&apos;s workspace in logs.
                    </p>
                </div>
            </div>
        </div>
    )
}
