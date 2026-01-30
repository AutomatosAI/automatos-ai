'use client'

import { useState } from 'react'
import { Search, Zap, ExternalLink } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'

const PROVIDER_FILTERS = [
  { id: 'all', name: 'All Providers' },
  { id: 'anthropic', name: 'Anthropic' },
  { id: 'openai', name: 'OpenAI' },
  { id: 'aiml', name: 'AIML' },
  { id: 'together', name: 'Together' },
]

export function MarketplaceLlmsTab() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedProvider, setSelectedProvider] = useState('all')

  const { data: llms = [], isLoading } = useQuery({
    queryKey: ['marketplaceLlms', selectedProvider, searchQuery],
    queryFn: async () => {
      const params = new URLSearchParams({
        type: 'llm',
        ...(selectedProvider !== 'all' && { category: selectedProvider }),
        ...(searchQuery && { search: searchQuery }),
      })
      const response = await fetch(`/api/marketplace/items?${params}`)
      if (!response.ok) throw new Error('Failed to fetch LLMs')
      return response.json()
    },
  })

  return (
    <div className="space-y-6">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
        <Input
          type="text"
          placeholder="Search LLM models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 bg-[#1a1a1a] border-gray-800 text-white placeholder:text-gray-500"
        />
      </div>

      <div className="flex flex-wrap gap-2">
        {PROVIDER_FILTERS.map((provider) => (
          <Button
            key={provider.id}
            variant={selectedProvider === provider.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedProvider(provider.id)}
            className={
              selectedProvider === provider.id
                ? 'bg-orange-500 hover:bg-orange-600 text-white'
                : 'border-gray-700 text-gray-300 hover:bg-gray-800'
            }
          >
            {provider.name}
          </Button>
        ))}
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-48 bg-gray-800 animate-pulse rounded-lg" />
          ))}
        </div>
      ) : llms.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <Zap className="w-12 h-12 mx-auto mb-4 text-gray-600" />
          <p>No LLMs found.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {llms.map((llm: any) => (
            <Card key={llm.id} className="bg-[#1a1a1a] border-gray-800 hover:border-orange-500/50 transition-all">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/30 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-purple-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-white line-clamp-1">{llm.name}</h3>
                    <p className="text-xs text-gray-500">{llm.category}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-gray-400 line-clamp-2">{llm.description}</p>
                <div className="flex flex-wrap gap-1">
                  {llm.tags.map((tag: string, idx: number) => (
                    <Badge key={idx} variant="outline" className="text-xs border-gray-700 text-gray-400">
                      {tag}
                    </Badge>
                  ))}
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-gray-800">
                  <span className="text-xs text-gray-500">
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
