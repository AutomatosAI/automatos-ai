'use client'

import { useState, useEffect } from 'react'
import { X, Download } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/components/ui/use-toast'
import type { MarketplaceItem } from './marketplace-homepage'

interface MarketplaceItemModalProps {
  itemId: number
  onClose: () => void
}

export function MarketplaceItemModal({
  itemId,
  onClose
}: MarketplaceItemModalProps) {
  const [item, setItem] = useState<MarketplaceItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [installing, setInstalling] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    fetchItem()
  }, [itemId])

  async function fetchItem() {
    try {
      const response = await fetch(`/api/marketplace/items/${itemId}`)
      if (response.ok) {
        const data = await response.json()
        setItem(data)
      }
    } catch (error) {
      console.error('Failed to fetch item details:', error)
      toast({
        title: 'Error',
        description: 'Failed to load item details',
        variant: 'destructive'
      })
    } finally {
      setLoading(false)
    }
  }

  async function handleInstall() {
    if (!item) return

    setInstalling(true)
    try {
      const response = await fetch(`/api/marketplace/items/${item.id}/install`, {
        method: 'POST'
      })

      if (response.ok) {
        const result = await response.json()
        toast({
          title: 'Success',
          description: result.message || 'Item installed successfully'
        })
        onClose()
      } else {
        throw new Error('Installation failed')
      }
    } catch (error) {
      console.error('Failed to install item:', error)
      toast({
        title: 'Error',
        description: 'Failed to install item',
        variant: 'destructive'
      })
    } finally {
      setInstalling(false)
    }
  }

  if (loading || !item) {
    return (
      <Dialog open={true} onOpenChange={onClose}>
        <DialogContent>
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded mb-4" />
            <div className="h-32 bg-gray-200 rounded" />
          </div>
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              {item.icon && <div className="text-4xl">{item.icon}</div>}
              <div>
                <DialogTitle className="text-2xl">{item.name}</DialogTitle>
                <p className="text-sm text-gray-500 mt-1">
                  by {item.creator_name} · v{item.version}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-6 w-6"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Description</h4>
            <p className="text-gray-600">{item.description}</p>
          </div>

          {item.category && (
            <div>
              <h4 className="font-semibold mb-2">Category</h4>
              <Badge variant="outline">{item.category}</Badge>
            </div>
          )}

          {item.tags.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Tags</h4>
              <div className="flex flex-wrap gap-2">
                {item.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center gap-4 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <Download className="h-4 w-4" />
              {item.install_count} installs
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleInstall} disabled={installing}>
            {installing ? 'Installing...' : 'Install'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
