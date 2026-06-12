'use client'

import React, { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload,
  X,
  HardDrive,
  Cloud,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { TeamMultiSelect } from './team-multi-select'

interface Provider {
  id: string
  name: string
  type: 'manual' | 'googledrive' | 'dropbox' | 'onedrive' | 'box'
  icon: React.ComponentType<any>
  color: string
  connected: boolean
  connectionId?: number
}

interface UploadProviderModalProps {
  open: boolean
  onClose: () => void
  providers: Provider[]
  onUpload: (files: FileList, providerId: string, connectionId?: number, teamAccess?: string[]) => Promise<void>
}

export function UploadProviderModal({
  open,
  onClose,
  providers,
  onUpload
}: UploadProviderModalProps) {
  const [selectedProviderId, setSelectedProviderId] = useState<string>('manual')
  const [teamAccess, setTeamAccess] = useState<string[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadComplete, setUploadComplete] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const selectedProvider = providers.find(p => p.id === selectedProviderId)

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0 || !selectedProvider) return

    setUploading(true)
    setUploadProgress(0)

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)

      await onUpload(files, selectedProvider.id, selectedProvider.connectionId, teamAccess.length > 0 ? teamAccess : undefined)

      clearInterval(progressInterval)
      setUploadProgress(100)
      setUploadComplete(true)

      // Auto-close after success
      setTimeout(() => {
        handleClose()
      }, 1500)
    } catch (error) {
      console.error('Upload error:', error)
      setUploading(false)
      setUploadProgress(0)
    }
  }

  const handleClose = () => {
    if (!uploading) {
      setSelectedProviderId('manual')
      setTeamAccess([])
      setUploadProgress(0)
      setUploadComplete(false)
      setUploading(false)
      onClose()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileUpload(e.target.files)
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files)
    }
  }

  const connectedProviders = providers.filter(p => p.connected)

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent size="md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Documents
          </DialogTitle>
          <DialogDescription>
            Choose where to upload your documents
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Provider Selection */}
          {!uploading && !uploadComplete && (
            <>
              <div className="space-y-3">
                <Label className="text-sm font-medium">Storage Destination</Label>
                <RadioGroup value={selectedProviderId} onValueChange={setSelectedProviderId}>
                  <div className="space-y-2">
                    {connectedProviders.map((provider) => {
                      const Icon = provider.icon
                      const isSelected = selectedProviderId === provider.id

                      return (
                        <div
                          key={provider.id}
                          className={`flex items-center space-x-3 p-4 rounded-lg border-2 transition-all cursor-pointer ${
                            isSelected
                              ? 'border-primary bg-primary/5'
                              : 'border-border hover:border-primary/50'
                          }`}
                          onClick={() => setSelectedProviderId(provider.id)}
                        >
                          <RadioGroupItem value={provider.id} id={provider.id} />
                          <div className="flex items-center gap-3 flex-1">
                            <div className={`w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center`}>
                              <Icon className={`w-5 h-5 ${provider.color}`} />
                            </div>
                            <div className="flex-1">
                              <Label htmlFor={provider.id} className="font-medium cursor-pointer">
                                {provider.name}
                              </Label>
                              <p className="text-xs text-muted-foreground">
                                {provider.type === 'manual'
                                  ? 'Upload directly to Automatos'
                                  : `Sync to your ${provider.name} account`}
                              </p>
                            </div>
                            {provider.connected && (
                              <Badge variant="outline" className="text-xs bg-success/10 text-success border-success/20">
                                <CheckCircle className="w-3 h-3 mr-1" />
                                Connected
                              </Badge>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </RadioGroup>
              </div>

              {/* Team Access */}
              <div className="space-y-2">
                <Label className="text-sm font-medium">Team Access</Label>
                <p className="text-xs text-muted-foreground">
                  Restrict visibility to specific teams. Leave empty for all teams.
                </p>
                {/* PRD-158 S2: team dropdown from /api/teams (no free-text). */}
                <TeamMultiSelect value={teamAccess} onChange={setTeamAccess} />
              </div>

              {/* File Drop Zone */}
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-220 ${
                  dragActive
                    ? 'border-primary bg-primary/5'
                    : 'border-border/50 hover:border-primary/50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.md,.xlsx,.csv,.json,.xml"
                  onChange={handleFileChange}
                  className="hidden"
                />

                <Upload className={`w-12 h-12 mx-auto mb-4 ${
                  dragActive ? 'text-primary' : 'text-muted-foreground'
                }`} />

                <h3 className="text-lg font-semibold mb-2">
                  {dragActive ? 'Drop files here' : 'Drag and drop files here'}
                </h3>
                <p className="text-muted-foreground mb-4">
                  Supports PDF, DOCX, TXT, MD, XLSX, CSV, JSON, XML
                </p>
                <Button
                  className="gradient-accent hover:opacity-90"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Choose Files
                </Button>
              </div>
            </>
          )}

          {/* Upload Progress */}
          {uploading && !uploadComplete && (
            <div className="space-y-4 py-8">
              <div className="flex items-center justify-center gap-3">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
                <span className="text-lg font-medium">Uploading to {selectedProvider?.name}...</span>
              </div>

              <div className="w-full bg-secondary rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-warning to-red-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>

              <p className="text-center text-sm text-muted-foreground">
                {uploadProgress}% complete
              </p>
            </div>
          )}

          {/* Upload Complete */}
          {uploadComplete && (
            <div className="space-y-4 py-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', duration: 0.5 }}
              >
                <CheckCircle className="w-16 h-16 mx-auto text-success" />
              </motion.div>

              <h3 className="text-xl font-semibold">Upload Complete!</h3>
              <p className="text-muted-foreground">
                Your documents have been uploaded to {selectedProvider?.name}
              </p>
            </div>
          )}

          {/* Actions */}
          {!uploading && !uploadComplete && (
            <div className="flex justify-end gap-3">
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
