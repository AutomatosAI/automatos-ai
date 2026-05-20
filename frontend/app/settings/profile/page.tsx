'use client'

import { useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { useUser } from '@clerk/nextjs'
import { User, Mail, Shield, Trash2, Upload, Save, X, CheckCircle, AlertCircle, Loader2, ArrowLeft, Briefcase, Copy, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useWorkspace } from '@/hooks/use-workspace'
import { PageHeader } from '@/components/shared'

/**
 * Custom Profile Page
 * 
 * Fully custom profile management page with:
 * - Personal information editing
 * - Avatar upload
 * - Email management
 * - Security settings
 * - Proper contrast and styling
 */

export default function ProfilePage() {
    const router = useRouter()
    const { user, isLoaded } = useUser()
    const [isEditing, setIsEditing] = useState(false)
    const [isSaving, setIsSaving] = useState(false)
    const [isUploadingAvatar, setIsUploadingAvatar] = useState(false)
    const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
    const [saveError, setSaveError] = useState<string | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        username: '',
    })
    const { workspaceId, loading: workspaceLoading } = useWorkspace()
    const [copiedWorkspaceId, setCopiedWorkspaceId] = useState(false)

    const handleCopyWorkspaceId = async () => {
        if (!workspaceId) return
        try {
            await navigator.clipboard.writeText(workspaceId)
            setCopiedWorkspaceId(true)
            setTimeout(() => setCopiedWorkspaceId(false), 2000)
        } catch (err) {
            console.error('Failed to copy workspace ID:', err)
        }
    }

    if (!isLoaded) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
            </div>
        )
    }

    if (!user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <p className="text-muted-foreground">Please sign in to view your profile.</p>
            </div>
        )
    }

    const handleAvatarClick = () => {
        fileInputRef.current?.click()
    }

    const handleAvatarUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        if (!file) return

        setIsUploadingAvatar(true)
        setSaveStatus('idle')
        try {
            await user.setProfileImage({ file })
            setSaveStatus('success')
            setTimeout(() => setSaveStatus('idle'), 3000)
        } catch (error) {
            console.error('Failed to upload avatar:', error)
            setSaveStatus('error')
            setTimeout(() => setSaveStatus('idle'), 3000)
        } finally {
            setIsUploadingAvatar(false)
        }
    }

    const handleSave = async () => {
        setIsSaving(true)
        setSaveStatus('idle')
        setSaveError(null)
        try {
            // Only update fields that have changed
            const updates: { firstName?: string; lastName?: string; username?: string } = {}

            if (formData.firstName !== (user.firstName || '')) {
                updates.firstName = formData.firstName
            }
            if (formData.lastName !== (user.lastName || '')) {
                updates.lastName = formData.lastName
            }
            if (formData.username !== (user.username || '')) {
                updates.username = formData.username
            }

            if (Object.keys(updates).length === 0) {
                setIsEditing(false)
                setSaveStatus('success')
                setTimeout(() => setSaveStatus('idle'), 3000)
                return
            }

            // Try the full update first. If Clerk rejects a single field
            // (typically username — disabled or taken), retry without it
            // so the other fields still save.
            try {
                await user.update(updates)
            } catch (err: any) {
                const clerkErrors: Array<{ code?: string; meta?: { paramName?: string }; longMessage?: string; message?: string }> =
                    err?.errors || []

                // If username was the offender and other fields exist, retry without username
                const usernameRejected = clerkErrors.some(
                    (e) => e.meta?.paramName === 'username' || e.code === 'form_identifier_exists' || e.code === 'form_param_unknown'
                )
                const otherFieldsToSave = Object.keys(updates).filter((k) => k !== 'username')

                if (usernameRejected && otherFieldsToSave.length > 0) {
                    const { username: _omit, ...rest } = updates
                    await user.update(rest)
                    const detail = clerkErrors.find((e) => e.meta?.paramName === 'username')?.longMessage
                        || 'Username could not be saved (already taken or disabled in Clerk). Other fields saved.'
                    setSaveError(detail)
                    setSaveStatus('error')
                    setTimeout(() => { setSaveStatus('idle'); setSaveError(null) }, 5000)
                    setIsEditing(false)
                    return
                }

                throw err
            }

            setIsEditing(false)
            setSaveStatus('success')
            setTimeout(() => setSaveStatus('idle'), 3000)
        } catch (error: any) {
            const clerkErrors = error?.errors as Array<{ longMessage?: string; message?: string }> | undefined
            const detail = clerkErrors?.[0]?.longMessage || clerkErrors?.[0]?.message || error?.message
            console.error('Failed to update profile:', error)
            setSaveError(detail || null)
            setSaveStatus('error')
            setTimeout(() => { setSaveStatus('idle'); setSaveError(null) }, 5000)
        } finally {
            setIsSaving(false)
        }
    }

    return (
        <div className="min-h-screen py-8 px-4">
            <div className="max-w-4xl mx-auto">
                {/* Page Header */}
                <div className="mb-8">
                    <PageHeader
                        title="Profile"
                        titleAccent="Settings"
                        eyebrow="Account · personal"
                        lede="Your name, email, role, and how you authenticate. Workspace-level settings live one level up under Workspace settings."
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

                {/* Success/Error Message */}
                {saveStatus !== 'idle' && (
                    <div className={`mb-6 p-4 rounded-lg border flex items-center space-x-3 ${saveStatus === 'success'
                        ? 'bg-success/10 border-success/30 text-success'
                        : 'bg-destructive/10 border-destructive/30 text-destructive'
                        }`}>
                        {saveStatus === 'success' ? (
                            <CheckCircle className="w-5 h-5" />
                        ) : (
                            <AlertCircle className="w-5 h-5" />
                        )}
                        <span>
                            {saveStatus === 'success'
                                ? 'Profile updated successfully!'
                                : saveError || 'Failed to update profile. Please try again.'}
                        </span>
                    </div>
                )}

                {/* Profile Card */}
                <div className="glass-card rounded-2xl border border-border p-8 mb-6">
                    {/* Avatar Section */}
                    <div className="flex items-start justify-between mb-8 pb-8 border-b border-border">
                        <div className="flex items-center space-x-6">
                            {user.imageUrl ? (
                                <img
                                    src={user.imageUrl}
                                    alt={user.fullName || 'User'}
                                    className="w-24 h-24 rounded-full border-4 border-primary/30"
                                />
                            ) : (
                                <div className="w-24 h-24 rounded-full bg-primary flex items-center justify-center">
                                    <User className="w-12 h-12 text-foreground" />
                                </div>
                            )}
                            <div>
                                <h2 className="text-2xl font-bold text-foreground mb-1">
                                    {user.fullName || 'User'}
                                </h2>
                                <p className="text-muted-foreground mb-3">
                                    {user.primaryEmailAddress?.emailAddress}
                                </p>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    onChange={handleAvatarUpload}
                                    className="hidden"
                                />
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={handleAvatarClick}
                                    disabled={isUploadingAvatar}
                                    className="border-primary/30 text-primary hover:bg-primary/10 hover:text-primary/80 disabled:opacity-50"
                                >
                                    {isUploadingAvatar ? (
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                    ) : (
                                        <Upload className="w-4 h-4 mr-2" />
                                    )}
                                    {isUploadingAvatar ? 'Uploading...' : 'Change Avatar'}
                                </Button>
                            </div>
                        </div>

                        {!isEditing ? (
                            <Button
                                onClick={() => {
                                    setFormData({
                                        firstName: user.firstName || '',
                                        lastName: user.lastName || '',
                                        username: user.username || '',
                                    })
                                    setIsEditing(true)
                                }}
                                className="bg-primary hover:bg-primary/90 text-primary-foreground"
                            >
                                Edit Profile
                            </Button>
                        ) : (
                            <div className="flex space-x-2">
                                <Button
                                    onClick={handleSave}
                                    disabled={isSaving}
                                    className="bg-primary hover:bg-primary/90 text-primary-foreground disabled:opacity-50"
                                >
                                    {isSaving ? (
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                    ) : (
                                        <Save className="w-4 h-4 mr-2" />
                                    )}
                                    {isSaving ? 'Saving...' : 'Save'}
                                </Button>
                                <Button
                                    onClick={() => setIsEditing(false)}
                                    variant="outline"
                                    className="border-border text-foreground hover:bg-secondary"
                                >
                                    <X className="w-4 h-4 mr-2" />
                                    Cancel
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* Personal Information */}
                    <div className="space-y-6">
                        <div>
                            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                                <User className="w-5 h-5 mr-2 text-primary" />
                                Personal Information
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-2">
                                        First Name
                                    </label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={formData.firstName}
                                            onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                                            className="w-full px-4 py-2 bg-secondary/50 border border-border rounded-lg text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
                                            placeholder="Enter first name"
                                        />
                                    ) : (
                                        <p className="px-4 py-2 bg-secondary/30 border border-border rounded-lg text-foreground">
                                            {user.firstName || 'Not set'}
                                        </p>
                                    )}
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-2">
                                        Last Name
                                    </label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={formData.lastName}
                                            onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                                            className="w-full px-4 py-2 bg-secondary/50 border border-border rounded-lg text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
                                            placeholder="Enter last name"
                                        />
                                    ) : (
                                        <p className="px-4 py-2 bg-secondary/30 border border-border rounded-lg text-foreground">
                                            {user.lastName || 'Not set'}
                                        </p>
                                    )}
                                </div>
                                <div className="md:col-span-2">
                                    <label className="block text-sm font-medium text-foreground mb-2">
                                        Username
                                    </label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={formData.username}
                                            onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                                            className="w-full px-4 py-2 bg-secondary/50 border border-border rounded-lg text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
                                            placeholder="Enter username"
                                        />
                                    ) : (
                                        <p className="px-4 py-2 bg-secondary/30 border border-border rounded-lg text-foreground">
                                            {user.username || 'Not set'}
                                        </p>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Email Addresses */}
                        <div className="pt-6 border-t border-border">
                            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                                <Mail className="w-5 h-5 mr-2 text-primary" />
                                Email Addresses
                            </h3>
                            <div className="space-y-3">
                                {user.emailAddresses.map((email) => (
                                    <div
                                        key={email.id}
                                        className="flex items-center justify-between px-4 py-3 bg-secondary/30 border border-border rounded-lg"
                                    >
                                        <div className="flex items-center space-x-3">
                                            <Mail className="w-4 h-4 text-muted-foreground" />
                                            <span className="text-foreground">{email.emailAddress}</span>
                                            {email.id === user.primaryEmailAddressId && (
                                                <span className="px-2 py-0.5 text-xs font-medium bg-primary/20 text-primary border border-primary/30 rounded">
                                                    Primary
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Workspace */}
                        <div className="pt-6 border-t border-border">
                            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                                <Briefcase className="w-5 h-5 mr-2 text-primary" />
                                Workspace
                            </h3>
                            <div>
                                <label className="block text-sm font-medium text-foreground mb-2">
                                    Workspace ID
                                </label>
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
                                    Include this ID when reporting an issue — it helps support trace your workspace in logs.
                                </p>
                            </div>
                        </div>

                        {/* Security */}
                        <div className="pt-6 border-t border-border">
                            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                                <Shield className="w-5 h-5 mr-2 text-primary" />
                                Security
                            </h3>
                            <div className="space-y-3">
                                <Button
                                    variant="outline"
                                    className="w-full justify-start border-border text-foreground hover:bg-secondary hover:text-foreground"
                                >
                                    Change Password
                                </Button>
                                <Button
                                    variant="outline"
                                    className="w-full justify-start border-border text-foreground hover:bg-secondary hover:text-foreground"
                                >
                                    Enable Two-Factor Authentication
                                </Button>
                            </div>
                        </div>

                        {/* Danger Zone */}
                        <div className="pt-6 border-t border-red-900/30">
                            <h3 className="text-lg font-semibold text-destructive mb-4 flex items-center">
                                <Trash2 className="w-5 h-5 mr-2" />
                                Danger Zone
                            </h3>
                            <Button
                                variant="outline"
                                className="border-destructive/30 text-destructive hover:bg-destructive/10 hover:text-destructive/80 hover:border-destructive/50"
                            >
                                <Trash2 className="w-4 h-4 mr-2" />
                                Delete Account
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
