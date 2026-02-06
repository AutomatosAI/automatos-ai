'use client'

import { useState } from 'react'
import { Search, Zap, ExternalLink, MoreVertical, Eye } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useUser } from '@clerk/nextjs'
import { toast } from 'sonner'

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
  const [approvingId, setApprovingId] = useState<number | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)
  const { user } = useUser()
  const queryClient = useQueryClient()

  // Admin check (same pattern as agents tab)
  const isAdmin = user?.emailAddresses?.[0]?.emailAddress?.includes('automatos.app') || false

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

  const handleApprove = async (e: React.MouseEvent, llmId: number) => {
    e.stopPropagation()
    setApprovingId(llmId)
    try {
      await apiClient.post(`/api/marketplace/items/${llmId}/approve`)
      toast.success('LLM approved and published to marketplace!')
      queryClient.invalidateQueries({ queryKey: ['marketplaceLlms'] })
    } catch (error: any) {
      toast.error('Failed to approve LLM', {
        description: error?.message || 'An error occurred',
      })
    } finally {
      setApprovingId(null)
    }
  }

  const handleDelete = async (e: React.MouseEvent, llmId: number) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this marketplace LLM?')) return
    setDeletingId(llmId)
    try {
      await apiClient.delete(`/api/marketplace/items/${llmId}`)
      toast.success('LLM removed from marketplace')
      queryClient.invalidateQueries({ queryKey: ['marketplaceLlms'] })
    } catch (error: any) {
      toast.error('Failed to delete LLM', {
        description: error?.message || 'An error occurred',
      })
    } finally {
      setDeletingId(null)
    }
  }

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
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
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
            <Card key={llm.id} className="glass-card card-glow hover:border-primary/20 transition-all duration-300 flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <Zap className="w-10 h-10 text-primary shrink-0" />
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-foreground line-clamp-1">{llm.name}</h3>
                      <p className="text-xs text-muted-foreground">{llm.category}</p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Open Provider
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">{llm.description}</p>
                <div className="flex flex-wrap gap-1">
                  {(llm.tags || []).map((tag: string, idx: number) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                <div className="pt-2 border-t border-secondary">
                  <span className="text-xs text-muted-foreground">
                    {llm.metadata?.context_window ? `${llm.metadata.context_window.toLocaleString()} tokens` : 'N/A'}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
