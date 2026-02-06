'use client'

import { useState } from 'react'
import { Zap, ExternalLink, Check, Trash2, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useQuery, useQueryClient } from '@tanstack/react-query'
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
            <Card key={llm.id} className="glass-card card-glow hover:border-primary/20 transition-all duration-300 flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/30 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-purple-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold line-clamp-1">{llm.name}</h3>
                      {isAdmin && !llm.is_approved && (
                        <Badge variant="outline" className="text-xs border-yellow-500/30 text-yellow-400">
                          Pending
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">{llm.category}</p>
                  </div>
                  {/* Admin Controls */}
                  {isAdmin && (
                    <div className="flex gap-1 flex-shrink-0">
                      {!llm.is_approved && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 w-7 p-0 border-green-500/30 text-green-400 hover:bg-green-500/10"
                          onClick={(e) => handleApprove(e, llm.id)}
                          disabled={approvingId === llm.id}
                          title="Approve"
                        >
                          {approvingId === llm.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 w-7 p-0 border-red-500/30 text-red-400 hover:bg-red-500/10"
                        onClick={(e) => handleDelete(e, llm.id)}
                        disabled={deletingId === llm.id}
                        title="Delete"
                      >
                        {deletingId === llm.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                      </Button>
                    </div>
                  )}
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
                <div className="flex items-center justify-between pt-2 border-t border-secondary mt-auto">
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
