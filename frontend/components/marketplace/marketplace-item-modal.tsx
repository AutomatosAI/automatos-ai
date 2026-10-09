'use client'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  ItemModalHeader,
  ItemModelSection,
  ItemSkillsSection,
  ItemToolsSection,
  useInstallMarketplaceDetailItem,
  useMarketplaceItemDetail,
} from './marketplace-item-sections'

interface MarketplaceItemModalProps {
  itemId: number
  /** 'agent' or 'recipe' (a playbook): agents and playbooks share id values. */
  itemType: string
  onClose: () => void
}

function ModalMessage({ onClose, children }: { onClose: () => void; children: React.ReactNode }) {
  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="glass-card card-glow border-border/50">{children}</DialogContent>
    </Dialog>
  )
}

export function MarketplaceItemModal({ itemId, itemType, onClose }: MarketplaceItemModalProps) {
  const { item, loading, error } = useMarketplaceItemDetail(itemId, itemType)
  const { installing, install } = useInstallMarketplaceDetailItem(itemType, onClose)

  if (error) {
    return (
      <ModalMessage onClose={onClose}>
        <div className="text-center py-8 space-y-4">
          <p className="text-[hsl(var(--destructive))]">{error}</p>
          <Button variant="outline" onClick={onClose}>Close</Button>
        </div>
      </ModalMessage>
    )
  }

  if (loading || !item) {
    return (
      <ModalMessage onClose={onClose}>
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-secondary/50 rounded" />
          <div className="h-32 bg-secondary/50 rounded" />
          <div className="h-24 bg-secondary/50 rounded" />
        </div>
      </ModalMessage>
    )
  }

  // Skills come in metadata (full details from the backend), or in dependencies.
  const skills = item.metadata?.skills || (item as any).dependencies?.skills || []

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden glass-card card-glow border-border/50">
        <ItemModalHeader item={item} installing={installing} onInstall={() => install(item)} />
        <div className="overflow-y-auto max-h-[calc(90vh-180px)] p-6 space-y-6">
          <div>
            <p className="text-muted-foreground leading-relaxed">
              {item.description || 'No description available'}
            </p>
          </div>
          <Separator />
          <ItemModelSection item={item} />
          <ItemSkillsSection skills={skills} />
          <ItemToolsSection item={item} />
        </div>
      </DialogContent>
    </Dialog>
  )
}
