'use client'

import { useState } from 'react'
import { MainLayout } from '@/components/layout/main-layout'
import { SemanticSearch } from '@/components/documents/semantic-search'
import { usePageAPI } from '@/hooks/use-page-api'
import type { SearchResult } from '@/hooks/use-semantic-search-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { FileText, Download, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export default function SearchPage() {
  usePageAPI('search')
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)

  const handleResultSelect = (result: SearchResult) => {
    setSelectedResult(result)
  }

  return (
    <MainLayout>
      <div className="p-6 max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold">Search Knowledge Base</h1>
          <p className="text-muted-foreground mt-1">
            Search across all your documents using semantic understanding
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Search Panel */}
          <div className="lg:col-span-2">
            <SemanticSearch
              context="documents"
              onResultSelect={handleResultSelect}
              standalone={true}
              maxResults={20}
            />
          </div>

          {/* Detail Panel */}
          <div className="lg:col-span-1">
            {selectedResult ? (
              <Card className="glass-card sticky top-6">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-blue-400" />
                    <CardTitle className="text-base truncate">
                      {selectedResult.source?.filename || 'Document'}
                    </CardTitle>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge className="text-green-400 bg-green-500/10 border-green-500/20">
                      {(selectedResult.similarity * 100).toFixed(0)}% Match
                    </Badge>
                    {selectedResult.source?.file_type && (
                      <Badge variant="outline">
                        {selectedResult.source.file_type.toUpperCase()}
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Content Preview */}
                  <div className="bg-muted/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                    <p className="text-sm whitespace-pre-wrap">
                      {selectedResult.preview || selectedResult.excerpt || selectedResult.content || ''}
                    </p>
                  </div>

                  {/* Metadata */}
                  <div className="space-y-2 text-xs text-muted-foreground">
                    {selectedResult.chunk_count && (
                      <div>{selectedResult.chunk_count} chunks in document</div>
                    )}
                    {selectedResult.source?.file_size != null && (
                      <div>{(selectedResult.source.file_size / 1024).toFixed(1)} KB</div>
                    )}
                    {selectedResult.source?.upload_date && (
                      <div>Uploaded: {new Date(selectedResult.source.upload_date).toLocaleDateString()}</div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    {selectedResult.document_id && (
                      <Button size="sm" variant="outline" asChild>
                        <a href={`/documents?doc=${selectedResult.document_id}`}>
                          <ExternalLink className="w-3 h-3 mr-1" />
                          Open Document
                        </a>
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="glass-card">
                <CardContent className="p-12 text-center">
                  <FileText className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-sm text-muted-foreground">
                    Select a search result to view details
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}
