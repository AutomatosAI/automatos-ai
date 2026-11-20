'use client'

/**
 * PRD-277 US-504 — the profile page in enterprise: read-only. The person's name and email come
 * from the organisation's identity provider at each sign-in, so they are changed there, and the
 * page says so. What it shows is the API's who-am-I (lib/enterprise/session-context.ts), drawn like
 * the local profile form's read-only fields (local-profile-form.tsx).
 */
import { useRouter } from 'next/navigation'
import { ArrowLeft, Briefcase, Mail, User } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { PageHeader } from '@/components/shared/page-header'
import { initialsFor } from '@/lib/initials'
import { PROFILE_MANAGED_NOTE } from '@/lib/enterprise/messages'
import { useEnterpriseSession } from '@/lib/enterprise/session-context'

const READONLY_CLASS =
  'w-full px-4 py-2 bg-secondary/30 border border-border rounded-lg text-muted-foreground cursor-not-allowed'

function ReadOnlyField({ id, label, value }: { id: string; label: string; value: string }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-foreground mb-2">{label}</label>
      <input id={id} value={value} readOnly aria-readonly="true" className={READONLY_CLASS} />
    </div>
  )
}

export function EnterpriseProfile() {
  const router = useRouter()
  const session = useEnterpriseSession()
  const person = session?.who?.person ?? null
  const workspaceId = session?.workspaceId ?? null
  const workspace = session?.who?.workspaces.find((held) => held.id === workspaceId) ?? null
  const name = person?.name ?? ''
  const email = person?.email ?? ''

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <PageHeader
            title="Profile"
            titleAccent="Settings"
            eyebrow="Managed by your organisation"
            lede="Your account is managed by your organisation's identity provider."
            actions={
              <Button variant="ghost" size="sm" onClick={() => router.back()}>
                <ArrowLeft className="w-4 h-4 mr-1" />
                Back
              </Button>
            }
          />
        </div>

        <section className="glass-card rounded-2xl border border-border p-8 mb-6 space-y-6">
          <div className="flex items-center space-x-6 pb-8 border-b border-border">
            <div
              aria-label="Avatar initials"
              className="w-24 h-24 rounded-full bg-primary flex items-center justify-center text-2xl font-semibold text-primary-foreground"
            >
              {initialsFor(name, email)}
            </div>
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-1">{name || email || 'Your account'}</h2>
              {email && <p className="text-muted-foreground">{email}</p>}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
              <User className="w-5 h-5 mr-2 text-primary" />
              Personal Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ReadOnlyField id="profile-name" label="Name" value={name} />
              <ReadOnlyField id="profile-email" label="Email" value={email} />
            </div>
            <p className="mt-2 text-xs text-muted-foreground flex items-center gap-1">
              <Mail className="w-3 h-3" />
              {PROFILE_MANAGED_NOTE}
            </p>
          </div>

          {workspace && (
            <div className="pt-6 border-t border-border">
              <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                <Briefcase className="w-5 h-5 mr-2 text-primary" />
                Workspace
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ReadOnlyField id="profile-workspace" label="Workspace" value={workspace.name} />
                <ReadOnlyField id="profile-workspace-role" label="Your role" value={workspace.role ?? ''} />
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
