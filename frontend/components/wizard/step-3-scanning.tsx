'use client'

import { Loader2, Search } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

interface Step3Props {
  domain: string
}

export function Step3Scanning({ domain }: Step3Props) {
  return (
    <Card className="bg-secondary/30 border-border/30">
      <CardContent className="py-12 text-center space-y-4">
        <div className="flex justify-center">
          <div className="relative">
            <Search className="w-12 h-12 text-primary" />
            <Loader2 className="w-12 h-12 absolute top-0 left-0 animate-spin text-primary/40" />
          </div>
        </div>
        <div>
          <div className="text-lg font-medium">Scanning {domain}…</div>
          <p className="text-sm text-muted-foreground mt-1">
            Discovering pages and detecting your business archetype. This usually takes 5-15 seconds.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
