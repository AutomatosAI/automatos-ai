'use client'

import React, { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { templateBlocksApi } from './api'
import type { BrandKit } from './types'

interface BrandKitDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved?: (kit: BrandKit) => void
}

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <Label className="text-xs">{label}</Label>
      <div className="flex items-center gap-2">
        <input
          type="color"
          value={/^#([0-9a-fA-F]{6})$/.test(value) ? value : '#1a1a2e'}
          onChange={(e) => onChange(e.target.value)}
          className="h-9 w-10 cursor-pointer rounded border bg-transparent"
          aria-label={label}
        />
        <Input value={value} onChange={(e) => onChange(e.target.value)} className="font-mono text-sm" />
      </div>
    </div>
  )
}

// Edit the workspace brand kit (PRD-167 S4). Drives {{brand.*}} and the PDF/DOCX palette.
export function BrandKitDialog({ open, onOpenChange, onSaved }: BrandKitDialogProps) {
  const [kit, setKit] = useState<BrandKit | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    templateBlocksApi
      .getBrandKit()
      .then(setKit)
      .catch(() => toast.error('Failed to load brand kit'))
  }, [open])

  const patch = (p: Partial<BrandKit>) => setKit((k) => (k ? { ...k, ...p } : k))
  const patchCompany = (p: Partial<BrandKit['company']>) =>
    setKit((k) => (k ? { ...k, company: { ...k.company, ...p } } : k))

  const save = async () => {
    if (!kit) return
    setSaving(true)
    try {
      const saved = await templateBlocksApi.updateBrandKit(kit)
      toast.success('Brand kit saved')
      onSaved?.(saved)
      onOpenChange(false)
    } catch (e: any) {
      toast.error(e?.message || 'Failed to save brand kit')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Brand Kit</DialogTitle>
        </DialogHeader>
        {!kit ? (
          <div className="py-8 text-center text-sm text-muted-foreground">Loading…</div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs">Brand name</Label>
                <Input value={kit.name} onChange={(e) => patch({ name: e.target.value })} />
              </div>
              <div>
                <Label className="text-xs">Tagline</Label>
                <Input value={kit.tagline} onChange={(e) => patch({ tagline: e.target.value })} />
              </div>
            </div>
            <div>
              <Label className="text-xs">Logo URL</Label>
              <Input value={kit.logo_url} onChange={(e) => patch({ logo_url: e.target.value })} placeholder="https://…/logo.png" />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <ColorField label="Primary" value={kit.primary_color} onChange={(v) => patch({ primary_color: v })} />
              <ColorField label="Secondary" value={kit.secondary_color} onChange={(v) => patch({ secondary_color: v })} />
              <ColorField label="Accent" value={kit.accent_color} onChange={(v) => patch({ accent_color: v })} />
            </div>
            <div>
              <Label className="text-xs">Font family</Label>
              <Input value={kit.font_family} onChange={(e) => patch({ font_family: e.target.value })} placeholder="Inter, system-ui, sans-serif" />
            </div>
            <div className="rounded-md border p-3">
              <p className="mb-2 text-xs font-medium text-muted-foreground">Company contact (drives {'{{company.*}}'})</p>
              <div className="grid grid-cols-2 gap-3">
                <Input placeholder="Address" value={kit.company.address} onChange={(e) => patchCompany({ address: e.target.value })} />
                <Input placeholder="Email" value={kit.company.email} onChange={(e) => patchCompany({ email: e.target.value })} />
                <Input placeholder="Phone" value={kit.company.phone} onChange={(e) => patchCompany({ phone: e.target.value })} />
                <Input placeholder="Website" value={kit.company.website} onChange={(e) => patchCompany({ website: e.target.value })} />
              </div>
            </div>
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={save} disabled={saving || !kit}>{saving ? 'Saving…' : 'Save'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
