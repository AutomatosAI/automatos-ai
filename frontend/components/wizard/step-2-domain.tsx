'use client'

import { Globe, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

interface Step2Props {
  domain: string
  onChange: (value: string) => void
  onBack: () => void
  onScan: () => void
  isScanning: boolean
}

export function Step2Domain({ domain, onChange, onBack, onScan, isScanning }: Step2Props) {
  const isValid = /^[a-z0-9.-]+\.[a-z]{2,}$/i.test(domain.trim())

  return (
    <Card className="bg-secondary/30 border-border/30">
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Globe className="w-4 h-4 text-primary" />
          What&apos;s your business website?
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          We&apos;ll scan your public site and use it to build your business profile. Nothing is saved
          outside your workspace.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label htmlFor="wizard-domain">Website domain</Label>
          <Input
            id="wizard-domain"
            placeholder="inbuilduk.com"
            value={domain}
            onChange={(e) => onChange(e.target.value)}
            className="bg-secondary/50 mt-1"
            autoFocus
          />
          <p className="text-xs text-muted-foreground mt-1">
            Just the domain — no https:// or paths.
          </p>
        </div>

        <div className="flex justify-between pt-2">
          <Button variant="ghost" onClick={onBack}>
            Back
          </Button>
          <Button onClick={onScan} disabled={!isValid || isScanning}>
            {isScanning ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Scanning…
              </>
            ) : (
              'Start Scan'
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
