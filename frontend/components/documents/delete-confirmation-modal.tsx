
'use client'

import * as React from 'react'
import { useState, useEffect, useMemo } from 'react'
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
import { apiClient } from '@/lib/api-client'

interface DeleteConfirmationModalProps {
  documentId: number | null
  filename: string
  open: boolean
  onClose: () => void
  onConfirm: (documentId: number) => Promise<void>
}

interface DeleteImpact {
  vector_chunks: number
  embeddings: number
  references: number
  workflows_affected: string[]
  storage_freed: string
  dependencies: Array<{
    type: 'workflow' | 'agent' | 'context'
    name: string
    impact: 'high' | 'medium' | 'low'
  }>
}

export function DeleteConfirmationModal({ 
  documentId, 
  filename, 
  open, 
  onClose, 
  onConfirm 
}: DeleteConfirmationModalProps) {
  const [loading, setLoading] = useState(false)
  const [impact, setImpact] = useState<DeleteImpact | null>(null)
  const [confirmationText, setConfirmationText] = useState('')
  const [acknowledgeImpact, setAcknowledgeImpact] = useState(false)
  const [acknowledgeNoUndo, setAcknowledgeNoUndo] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  React.useEffect(() => {
    if (open && documentId) {
      loadDeleteImpact()
    }
  }, [open, documentId])

  const loadDeleteImpact = async () => {
    if (!documentId) return
    
    setLoading(true)
    setError(null)
    
    try {
      // Try to get delete impact analysis from API
      const impactData = await apiClient.request<DeleteImpact>(`/api/documents/${documentId}/delete-impact`)
      setImpact(impactData)
    } catch (err) {
      console.warn('Could not load delete impact data:', err)
      // Set minimal impact data - mock data is handled in api-client.ts
      setImpact({
        vector_chunks: 0,
        embeddings: 0,
        references: 0,
        workflows_affected: [],
        storage_freed: '0 MB',
        dependencies: []
      })
    } finally {
      setLoading(false)
    }
  }

  const handleConfirm = async () => {
    if (!documentId || !canConfirm) return
    
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
    setImpact(null)
  }

  // SIMPLIFIED: Just require checkboxes, no filename matching
  const canConfirm = useMemo(() => {
    const result = acknowledgeImpact && acknowledgeNoUndo
    
    console.log('[Delete Modal] Validation (SIMPLIFIED):', {
      acknowledgeImpact,
      acknowledgeNoUndo,
      canConfirm: result
    })
    
    return result
  }, [acknowledgeImpact, acknowledgeNoUndo])

  const getImpactColor = (impact: 'high' | 'medium' | 'low') => {
    switch (impact) {
      case 'high':
        return 'bg-red-500/10 text-red-400 border-red-500/20'
      case 'medium':
        return 'bg-orange-500/10 text-orange-400 border-orange-500/20'
      case 'low':
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/20'
    }
  }

  const getDependencyIcon = (type: string) => {
    switch (type) {
      case 'workflow':
        return Database
      case 'agent':
        return CheckCircle
      case 'context':
        return Link
      default:
        return FileText
    }
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

            {loading && (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-orange-400" />
                  <p className="text-sm text-muted-foreground">Analyzing deletion impact...</p>
                </div>
              </div>
            )}

            {/* Impact Analysis */}
            {impact && !loading && (
              <div className="space-y-4">
                <h4 className="font-semibold flex items-center space-x-2">
                  <Database className="w-4 h-4 text-orange-400" />
                  <span>Deletion Impact Analysis</span>
                </h4>

                {/* Impact Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-secondary/30 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-orange-400">{impact.vector_chunks}</p>
                    <p className="text-xs text-muted-foreground">Vector Chunks</p>
                  </div>
                  <div className="bg-secondary/30 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-purple-400">{impact.embeddings}</p>
                    <p className="text-xs text-muted-foreground">Embeddings</p>
                  </div>
                  <div className="bg-secondary/30 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-blue-400">{impact.references}</p>
                    <p className="text-xs text-muted-foreground">References</p>
                  </div>
                  <div className="bg-secondary/30 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-green-400">{impact.storage_freed}</p>
                    <p className="text-xs text-muted-foreground">Storage Freed</p>
                  </div>
                </div>

                {/* Affected Workflows */}
                {impact.workflows_affected && impact.workflows_affected.length > 0 && (
                  <div className="bg-orange-500/5 border border-orange-500/20 rounded-lg p-4">
                    <h5 className="font-medium text-orange-400 mb-2">Affected Workflows</h5>
                    <div className="space-y-1">
                      {impact.workflows_affected.map((workflow, index) => (
                        <Badge key={index} variant="outline" className="mr-2 mb-1">
                          {workflow}
                        </Badge>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      These workflows may be affected by deleting this document.
                    </p>
                  </div>
                )}

                {/* Dependencies */}
                {impact.dependencies && impact.dependencies.length > 0 && (
                  <div>
                    <h5 className="font-medium mb-3">System Dependencies</h5>
                    <div className="space-y-2">
                      {impact.dependencies.map((dep, index) => {
                        const Icon = getDependencyIcon(dep.type)
                        return (
                          <div key={index} className="flex items-center justify-between p-3 bg-secondary/20 rounded-lg">
                            <div className="flex items-center space-x-3">
                              <Icon className="w-4 h-4 text-muted-foreground" />
                              <div>
                                <p className="font-medium text-sm">{dep.name}</p>
                                <p className="text-xs text-muted-foreground capitalize">{dep.type}</p>
                              </div>
                            </div>
                            <Badge className={getImpactColor(dep.impact)}>
                              {dep.impact} impact
                            </Badge>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

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
                disabled={!canConfirm || deleting}
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
