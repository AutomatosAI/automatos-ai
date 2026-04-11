'use client'

import { Loader2, FileText, Network, Database } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

interface Step5Props {
  pageCount: number
  isLoading: boolean
}

export function Step5Intake({ pageCount, isLoading }: Step5Props) {
  return (
    <Card className="bg-secondary/30 border-border/30">
      <CardContent className="py-12 space-y-6">
        <div className="text-center">
          <div className="text-lg font-medium">Reading {pageCount} pages…</div>
          <p className="text-sm text-muted-foreground mt-1">
            Scraping content, embedding to RAG, and building your knowledge graph.
          </p>
        </div>

        <div className="space-y-3 max-w-md mx-auto">
          <Stage icon={FileText} label="Scraping pages with Firecrawl" active={isLoading} />
          <Stage icon={Database} label="Ingesting to RAG (DocumentManager)" active={isLoading} />
          <Stage icon={Network} label="Building knowledge graph (Graphify)" active={isLoading} />
        </div>
      </CardContent>
    </Card>
  )
}

function Stage({
  icon: Icon,
  label,
  active,
}: {
  icon: typeof FileText
  label: string
  active: boolean
}) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-md bg-secondary/40 border border-border/30">
      {active ? (
        <Loader2 className="w-5 h-5 text-primary animate-spin" />
      ) : (
        <Icon className="w-5 h-5 text-primary" />
      )}
      <div className="text-sm">{label}</div>
    </div>
  )
}
