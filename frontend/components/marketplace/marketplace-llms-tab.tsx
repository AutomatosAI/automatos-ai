'use client'

import { useState } from 'react'
import { Search, Zap, ExternalLink } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

const PROVIDER_FILTERS = [
  { id: 'all', name: 'All Providers' },
  { id: 'anthropic', name: 'Anthropic' },
  { id: 'openai', name: 'OpenAI' },
  { id: 'aiml', name: 'AIML' },
  { id: 'together', name: 'Together' },
]

interface MarketplaceLlmsTabProps {
  searchQuery: string
}

export function MarketplaceLlmsTab({ searchQuery }: MarketplaceLlmsTabProps) {
  const [selectedProvider, setSelectedProvider] = useState('all')

  const { data: llms = [], isLoading } = useQuery({
    queryKey: ['marketplaceLlms', selectedProvider, searchQuery],
    queryFn: async () => {
      const params = new URLSearchParams({
        type: 'llm',
        ...(selectedProvider !== 'all' && { category: selectedProvider }),
        ...(searchQuery && { search: searchQuery }),
      })
      return apiClient.get(`/api/marketplace/items?${params}`)
    },
  })

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {PROVIDER_FILTERS.map((provider) => (
          <Button
            key={provider.id}
            variant={selectedProvider === provider.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedProvider(provider.id)}
            className={
              selectedProvider === provider.id
                ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }
          >
            {provider.name}
          </Button>
        ))}
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-48 glass-card animate-pulse" />
          ))}
        </div>
      ) : llms.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No LLMs found.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {llms.map((llm: any) => (
            <Card key={llm.id} className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/30 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-purple-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold line-clamp-1">{llm.name}</h3>
                    <p className="text-xs text-muted-foreground">{llm.category}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">{llm.description}</p>
                <div className="flex flex-wrap gap-1">
                  {llm.tags.map((tag: string, idx: number) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-secondary">
                  <span className="text-xs text-muted-foreground">
                    {llm.metadata?.context_window ? `${llm.metadata.context_window.toLocaleString()} tokens` : 'N/A'}
                  </span>
                  <Button size="sm" variant="outline" className="border-purple-500/30 text-purple-300 hover:bg-purple-500/10">
                    <ExternalLink className="w-3 h-3 mr-1" />
                    Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
