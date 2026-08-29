'use client'

import { useState } from 'react'
import {
  Download,
  Bot,
  Wrench,
  Brain,
  Puzzle,
  BookOpen,
  Cpu,
  Plug,
  CheckCircle,
  FileText,
  Loader2,
  AlertCircle,
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { apiClient } from '@/lib/api-client'
import { toast as sonnerToast } from 'sonner'
import {
  MarketplacePackage,
  PackageInstallResult,
  PackageMember,
  MEMBER_TYPE_LABELS,
  groupMembersByType,
  memberLabel,
  connectLabel,
  reportLabel,
} from './package-types'

interface PackageDetailModalProps {
  pkg: MarketplacePackage
  onClose: () => void
  /** Optional install override (tests inject a stub; prod hits the install route). */
  onInstalled?: (result: PackageInstallResult) => void
}

const TYPE_ICONS: Record<string, any> = {
  agent: Bot,
  playbook: BookOpen,
  skill: Brain,
  tool: Wrench,
  plugin: Puzzle,
  llm: Cpu,
}

function typeLabel(type: string, count: number): string {
  const labels = MEMBER_TYPE_LABELS[type]
  if (!labels) return `${count} ${type}`
  return `${labels.plural} · ${count}`
}

export function PackageDetailModal({ pkg, onClose, onInstalled }: PackageDetailModalProps) {
  const [installing, setInstalling] = useState(false)
  const [result, setResult] = useState<PackageInstallResult | null>(null)

  const grouped = groupMembersByType(pkg.members || [])
  const manifest = pkg.setup_manifest || {}
  const requiredConnects = manifest.required_connects || []
  const reports = manifest.report_templates || []

  async function handleInstall() {
    setInstalling(true)
    try {
      const data: PackageInstallResult = await apiClient.post(
        `/api/marketplace/packages/${pkg.slug}/install`,
      )
      setResult(data)

      if (data.success) {
        sonnerToast.success('Package installed', {
          description:
            data.message || `${pkg.name} added — ${data.added_count ?? 0} registered.`,
        })
        onInstalled?.(data)
      } else if (data.over_quota) {
        sonnerToast('Let’s pick the right plan', { description: data.message })
      } else if (data.onboarding_restricted) {
        sonnerToast('One package during onboarding', { description: data.message })
      } else {
        sonnerToast.error('Install failed', { description: data.message || data.error })
      }
    } catch (error: any) {
      sonnerToast.error('Install failed', {
        description: error?.message || 'Could not install this package. Please try again.',
      })
    } finally {
      setInstalling(false)
    }
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent
        className="max-w-3xl max-h-[90vh] overflow-hidden glass-card card-glow border-border/50"
        data-testid="package-detail-modal"
      >
        <DialogHeader className="border-b border-border/30 pb-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <DialogTitle className="text-2xl font-bold">{pkg.name}</DialogTitle>
              <div className="flex flex-wrap items-center gap-2 mt-2">
                {(pkg.vertical_tags || []).map((tag) => (
                  <Badge
                    key={tag}
                    variant="outline"
                    className="text-xs border-primary/30 text-primary"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
            <Button onClick={handleInstall} disabled={installing} className="flex-shrink-0">
              {installing ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              {installing ? 'Installing…' : 'Install package'}
            </Button>
          </div>
        </DialogHeader>

        <div className="overflow-y-auto max-h-[calc(90vh-180px)] p-6 space-y-6">
          {/* Description */}
          <p className="text-muted-foreground leading-relaxed">
            {pkg.description || 'No description available'}
          </p>

          <Separator />

          {/* What's included — members grouped by type */}
          <div data-testid="package-members">
            <h3 className="text-lg font-semibold mb-4">What’s included</h3>
            <div className="space-y-5">
              {grouped.map(({ type, members }) => {
                const Icon = TYPE_ICONS[type] || Puzzle
                return (
                  <div key={type} data-testid={`member-group-${type}`}>
                    <div className="flex items-center gap-2 mb-2 text-sm font-medium text-foreground">
                      <Icon className="w-4 h-4 text-primary" />
                      <span>{typeLabel(type, members.length)}</span>
                    </div>
                    <div className="space-y-2 pl-6">
                      {members.map((m: PackageMember, idx: number) => (
                        <div
                          key={`${type}-${idx}`}
                          className="bg-secondary/30 border border-border/30 rounded-lg p-3"
                        >
                          <p className="font-medium text-sm">{memberLabel(m)}</p>
                          {m.description && (
                            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                              {m.description}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
              {grouped.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  This package has no listed members yet.
                </p>
              )}
            </div>
          </div>

          {/* Setup summary — what gets connected + what reports you'll get */}
          {(requiredConnects.length > 0 || reports.length > 0) && (
            <>
              <Separator />
              <div data-testid="package-setup-summary">
                <h3 className="text-lg font-semibold mb-4">Setup</h3>

                {requiredConnects.length > 0 && (
                  <div className="mb-4">
                    <div className="flex items-center gap-2 mb-2 text-sm font-medium">
                      <Plug className="w-4 h-4 text-primary" />
                      <span>Apps to connect</span>
                    </div>
                    <ul className="space-y-1 pl-6">
                      {requiredConnects.map((c, idx) => (
                        <li key={idx} className="text-sm text-muted-foreground">
                          {connectLabel(c)}
                          {typeof c !== 'string' && c.note && (
                            <span className="block text-xs opacity-80">{c.note}</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {reports.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-2 text-sm font-medium">
                      <FileText className="w-4 h-4 text-primary" />
                      <span>Reports you’ll get</span>
                    </div>
                    <ul className="space-y-1 pl-6">
                      {reports.map((r, idx) => (
                        <li key={idx} className="text-sm text-muted-foreground">
                          {reportLabel(r)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Install result — the returned manifest (or the honest non-install copy) */}
          {result && (
            <>
              <Separator />
              <div
                data-testid="install-result"
                className="bg-secondary/30 border border-border/30 rounded-lg p-4"
              >
                {result.success ? (
                  <>
                    <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-[hsl(var(--success))]">
                      <CheckCircle className="w-4 h-4" />
                      <span>
                        Installed — {result.added_count ?? result.registrations?.length ?? 0}{' '}
                        registered to your workspace
                      </span>
                    </div>
                    {result.registrations && result.registrations.length > 0 && (
                      <ul className="space-y-1 mb-3" data-testid="install-registrations">
                        {result.registrations.map((r, idx) => (
                          <li
                            key={idx}
                            className="flex items-center justify-between text-xs text-muted-foreground"
                          >
                            <span>
                              {r.name}{' '}
                              <span className="opacity-70">({r.type})</span>
                            </span>
                            <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                              {r.status}
                            </Badge>
                          </li>
                        ))}
                      </ul>
                    )}
                    {result.required_connects && result.required_connects.length > 0 && (
                      <div>
                        <p className="text-xs font-medium mb-1">Next: connect these apps</p>
                        <ul className="space-y-1">
                          {result.required_connects.map((c, idx) => (
                            <li key={idx} className="text-xs text-muted-foreground">
                              {c.app_name}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex items-start gap-2 text-sm text-muted-foreground">
                    <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-[hsl(var(--warning))]" />
                    <span>
                      {result.message || result.error || 'Nothing was installed.'}
                      {result.plan_recommendation && (
                        <span className="block mt-1 text-xs">
                          Recommended plan: {result.plan_recommendation}
                        </span>
                      )}
                    </span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
