'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  AlertTriangle, 
  Trash2, 
  CheckCircle,
  Info
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { useAgent, useDeleteAgent } from '@/hooks/use-agent-api'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'

interface AgentConfirmDeleteModalProps {
  agentId: number | null
  open: boolean
  onClose: () => void
  onDeleted?: () => void
}

export function AgentConfirmDeleteModal({ 
  agentId, 
  open, 
  onClose,
  onDeleted
}: AgentConfirmDeleteModalProps) {
  const [isConfirmed, setIsConfirmed] = useState(false)
  
  // Use real API hooks
  const { data: agent, isLoading } = useAgent(agentId?.toString() || '')
  const deleteAgentMutation = useDeleteAgent()

  const handleDelete = async () => {
    if (!agentId || !isConfirmed) return
    
    try {
      await deleteAgentMutation.mutateAsync(agentId.toString())
      
      // Notify parent component
      if (onDeleted) {
        onDeleted()
      }
      
      setIsConfirmed(false)
      onClose()
    } catch (error) {
      console.error('Error deleting agent:', error)
    }
  }

  if (!open || !agentId) return null

  return (
    <AnimatePresence>
      <motion.div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }} 
        onClick={onClose}
      />
      <motion.div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4" 
        initial={{ opacity: 0, scale: 0.95 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-md">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3 text-xl">
              <Trash2 className="w-5 h-5 text-red-500" />
              <span>Confirm Delete</span>
            </CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </CardHeader>
          
          <CardContent className="pt-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-6">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start space-x-3">
                  <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-red-500">Warning: Permanent Action</h3>
                    <p className="text-sm text-muted-foreground">
                      You are about to delete <strong>{agent?.name}</strong>. This action cannot be undone.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-muted-foreground">
                      This will remove all configuration, skills associations, and historical data for this agent.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3 pt-4">
                  <Checkbox 
                    id="confirm-delete" 
                    checked={isConfirmed} 
                    onCheckedChange={(checked) => setIsConfirmed(checked === true)}
                  />
                  <Label htmlFor="confirm-delete" className="text-sm">
                    I understand that this action is permanent and cannot be reversed
                  </Label>
                </div>
              </div>
            )}
          </CardContent>
          
          <CardFooter className="flex justify-between border-t border-border/30 p-4">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={handleDelete}
              disabled={!isConfirmed || isLoading || deleteAgentMutation.isPending}
            >
              {deleteAgentMutation.isPending ? (
                <>
                  <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></span>
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Agent
                </>
              )}
            </Button>
          </CardFooter>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
