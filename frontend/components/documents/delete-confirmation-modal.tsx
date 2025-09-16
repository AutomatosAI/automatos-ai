'use client'

import * as React from 'react'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Trash2, 
  AlertTriangle, 
  FileText,
  Database,
  Link,
  Loader2,
  CheckCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

interface DeleteConfirmationModalProps {
  documentId: number | null
  filename: string
  open: boolean
  onClose: () => void
  onConfirm: (documentId: number) => Promise<void>
}

export function DeleteConfirmationModal({ 
  documentId, 
  filename, 
  open, 
  onClose, 
  onConfirm 
}: DeleteConfirmationModalProps) {
  const [loading, setLoading] = useState(false)
  const [confirmationText, setConfirmationText] = useState('')
  const [acknowledgeImpact, setAcknowledgeImpact] = useState(false)
  const [acknowledgeNoUndo, setAcknowledgeNoUndo] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  const handleConfirm = async () => {
    if (!documentId || !canConfirm()) return
    
    setDeleting(true)
    setError(null)
    
    try {
      await onConfirm(documentId)
      onClose()
      resetState()
    } catch (err) {
      console.error('Delete error:', err)
      setError(err instanceof Error ? err.message : 'Failed to delete document')
    } finally {
      setDeleting(false)
    }
  }

  const resetState = () => {
    setConfirmationText('')
    setAcknowledgeImpact(false)
    setAcknowledgeNoUndo(false)
    setError(null)
  }

  const canConfirm = () => {
    return (
      confirmationText.toLowerCase() === filename.toLowerCase() &&
      acknowledgeImpact &&
      acknowledgeNoUndo
    )
  }

  if (!open || !documentId) return null

  return (
    <AnimatePresence>
      <motion.div 
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }} 
      />
      <motion.div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4" 
        initial={{ opacity: 0, scale: 0.95 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-2xl max-h-[90vh] overflow-hidden">
          <CardHeader className="bg-red-500/5 border-b border-red-500/20">
            <CardTitle className="flex items-center space-x-2 text-red-400">
              <AlertTriangle className="w-6 h-6" />
              <span>Confirm Document Deletion</span>
            </CardTitle>
            <Button 
              variant="ghost" 
              size="icon" 
              className="absolute right-4 top-4"
              onClick={onClose}
              disabled={deleting}
            >
              <X className="w-4 h-4" />
            </Button>
          </CardHeader>
          
          <CardContent className="overflow-y-auto p-6 space-y-6">
            {/* Warning Message */}
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-red-400 mb-1">
                    You are about to permanently delete this document
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    This action cannot be undone. The document and all associated data will be permanently removed from the system.
                  </p>
                </div>
              </div>
            </div>

            {/* Document Info */}
            <div className="bg-secondary/30 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-lg bg-red-500/20 flex items-center justify-center">
                  <FileText className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg">{filename}</h3>
                  <p className="text-sm text-muted-foreground">Document ID: #{documentId}</p>
                </div>
              </div>
            </div>

            {/* Impact Analysis */}
            <div className="space-y-4">
              <h4 className="font-semibold flex items-center space-x-2">
                <Database className="w-4 h-4 text-orange-400" />
                <span>Deletion Impact</span>
              </h4>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-secondary/30 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-orange-400">45</p>
                  <p className="text-xs text-muted-foreground">Vector Chunks</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-purple-400">128</p>
                  <p className="text-xs text-muted-foreground">Embeddings</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-400">0</p>
                  <p className="text-xs text-muted-foreground">References</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-green-400">2.4 MB</p>
                  <p className="text-xs text-muted-foreground">Storage Freed</p>
                </div>
              </div>
            </div>

            {/* Confirmation Requirements */}
            <div className="space-y-4 border-t border-border/30 pt-6">
              <h4 className="font-semibold text-red-400">Confirmation Required</h4>
              
              {/* Type filename confirmation */}
              <div className="space-y-2">
                <Label htmlFor="confirmation">
                  Type the filename to confirm deletion: <span className="font-mono text-sm">{filename}</span>
                </Label>
                <Input
                  id="confirmation"
                  value={confirmationText}
                  onChange={(e) => setConfirmationText(e.target.value)}
                  placeholder={filename}
                  className="bg-secondary/50 border-border/50"
                  disabled={deleting}
                />
              </div>

              {/* Acknowledgment checkboxes */}
              <div className="space-y-3">
                <div className="flex items-start space-x-2">
                  <Checkbox 
                    id="acknowledge-impact"
                    checked={acknowledgeImpact}
                    onCheckedChange={(checked) => setAcknowledgeImpact(checked === true)}
                    disabled={deleting}
                  />
                  <Label htmlFor="acknowledge-impact" className="text-sm leading-5">
                    I understand the impact of this deletion and accept responsibility for any system disruptions.
                  </Label>
                </div>
                
                <div className="flex items-start space-x-2">
                  <Checkbox 
                    id="acknowledge-no-undo"
                    checked={acknowledgeNoUndo}
                    onCheckedChange={(checked) => setAcknowledgeNoUndo(checked === true)}
                    disabled={deleting}
                  />
                  <Label htmlFor="acknowledge-no-undo" className="text-sm leading-5">
                    I understand that this action cannot be undone and the document will be permanently deleted.
                  </Label>
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3 border-t border-border/30 pt-6">
              <Button 
                variant="outline" 
                onClick={onClose}
                disabled={deleting}
              >
                Cancel
              </Button>
              <Button 
                variant="destructive"
                onClick={handleConfirm}
                disabled={!canConfirm() || deleting}
                className="bg-red-600 hover:bg-red-700"
              >
                {deleting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete Document
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
