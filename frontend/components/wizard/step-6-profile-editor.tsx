'use client'

import { useState, useEffect } from 'react'
import { Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import type { BusinessProfilePayload, ScrapeResponse } from '@/hooks/use-wizard-api'

interface Step6Props {
  profile: BusinessProfilePayload
  scrape: ScrapeResponse | null
  onSave: (patch: Partial<BusinessProfilePayload>) => Promise<void> | void
  isSaving: boolean
  onBack: () => void
  onGeneratePlan: () => void
  isGenerating: boolean
}

export function Step6ProfileEditor({
  profile,
  scrape,
  onSave,
  isSaving,
  onBack,
  onGeneratePlan,
  isGenerating,
}: Step6Props) {
  const [companyName, setCompanyName] = useState(profile.company_name || '')
  const [sectors, setSectors] = useState((profile.sectors || []).join(', '))
  const [standards, setStandards] = useState((profile.standards || []).join(', '))
  const [voiceNotes, setVoiceNotes] = useState(profile.voice_notes || '')

  useEffect(() => {
    setCompanyName(profile.company_name || '')
    setSectors((profile.sectors || []).join(', '))
    setStandards((profile.standards || []).join(', '))
    setVoiceNotes(profile.voice_notes || '')
  }, [profile])

  const save = async () => {
    await onSave({
      company_name: companyName.trim() || null,
      sectors: sectors.split(',').map(s => s.trim()).filter(Boolean),
      standards: standards.split(',').map(s => s.trim()).filter(Boolean),
      voice_notes: voiceNotes.trim() || null,
    })
  }

  const findings = profile.quality_findings || {}
  const errors = findings.errors || []
  const notes = findings.notes || []

  return (
    <div className="space-y-4">
      {/* Intake summary */}
      {scrape && (
        <Card className="bg-secondary/30 border-border/30">
          <CardContent className="py-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="w-6 h-6 text-primary" />
              <div className="flex-1 text-sm">
                <span className="font-medium">{scrape.pages_scraped}</span> pages scraped ·{' '}
                <span className="font-medium">{scrape.documents_ingested}</span> documents ingested
                {scrape.pages_failed > 0 && (
                  <span className="text-destructive">
                    {' '}
                    · {scrape.pages_failed} failed
                  </span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Profile editor */}
      <Card className="bg-secondary/30 border-border/30">
        <CardHeader>
          <CardTitle className="text-base">Your business profile</CardTitle>
          <p className="text-sm text-muted-foreground">
            We extracted these details from your site. Edit anything that&apos;s wrong.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="profile-company">Company name</Label>
            <Input
              id="profile-company"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              className="bg-secondary/50 mt-1"
              placeholder="Acme Ltd"
            />
          </div>
          <div>
            <Label htmlFor="profile-sectors">Sectors served (comma-separated)</Label>
            <Input
              id="profile-sectors"
              value={sectors}
              onChange={(e) => setSectors(e.target.value)}
              className="bg-secondary/50 mt-1"
              placeholder="industrial, commercial, education"
            />
          </div>
          <div>
            <Label htmlFor="profile-standards">Standards / certifications (comma-separated)</Label>
            <Input
              id="profile-standards"
              value={standards}
              onChange={(e) => setStandards(e.target.value)}
              className="bg-secondary/50 mt-1"
              placeholder="EN 12101-8, BS 5839"
            />
          </div>
          <div>
            <Label htmlFor="profile-voice">Voice notes</Label>
            <Textarea
              id="profile-voice"
              value={voiceNotes}
              onChange={(e) => setVoiceNotes(e.target.value)}
              className="bg-secondary/50 mt-1"
              rows={4}
              placeholder="How does your brand talk? Any tone preferences?"
            />
          </div>

          {profile.brands && profile.brands.length > 0 && (
            <div>
              <Label>Brands detected ({profile.brands.length})</Label>
              <div className="flex flex-wrap gap-1 mt-2">
                {profile.brands.slice(0, 12).map((b: any, i: number) => (
                  <Badge key={i} variant="outline">
                    {b.brand_name || b.name || 'Unknown'}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quality findings */}
      {(errors.length > 0 || notes.length > 0) && (
        <Card className="bg-secondary/30 border-border/30">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              Quality findings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            {errors.map((e, i) => (
              <div key={`e${i}`} className="text-destructive">
                · {e}
              </div>
            ))}
            {notes.map((n, i) => (
              <div key={`n${i}`} className="text-muted-foreground">
                · {n}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="flex justify-between">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
        <div className="flex gap-2">
          <Button variant="outline" onClick={save} disabled={isSaving}>
            {isSaving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            Save Edits
          </Button>
          <Button onClick={onGeneratePlan} disabled={isGenerating}>
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Launching mission…
              </>
            ) : (
              'Launch Mission Zero →'
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}
