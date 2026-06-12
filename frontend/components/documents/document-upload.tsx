
'use client'

import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  Upload, 
  FileText, 
  FileImage, 
  FileVideo, 
  FileAudio, 
  Archive,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  Plus,
  Settings,
  Tag,
  FolderOpen
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { toast } from 'sonner'

// API hooks
import { useUploadDocument } from '@/hooks/use-document-api'

interface FileUpload {
  id: string
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  progress: number
  error?: string
  metadata: {
    name: string
    description: string
    category: string
    tags: string[]
    team_access: string[]
    auto_process: boolean
    auto_index: boolean
  }
}

const fileTypeIcons: Record<string, any> = {
  'application/pdf': FileText,
  'image/': FileImage,
  'video/': FileVideo,
  'audio/': FileAudio,
  'application/zip': Archive,
  'application/x-rar': Archive,
  'application/x-7z': Archive,
  'text/': FileText,
  default: FileText
}

const maxFileSize = 100 * 1024 * 1024 // 100MB
const acceptedFileTypes = {
  'application/pdf': ['.pdf'],
  'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'],
  'video/*': ['.mp4', '.avi', '.mov', '.wmv', '.flv'],
  'audio/*': ['.mp3', '.wav', '.flac', '.aac'],
  'text/*': ['.txt', '.md', '.csv'],
  'application/msword': ['.doc'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/vnd.ms-excel': ['.xls'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.ms-powerpoint': ['.ppt'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'application/zip': ['.zip'],
  'application/x-rar-compressed': ['.rar'],
  'application/x-7z-compressed': ['.7z']
}

interface DocumentUploadProps {
  onUploadSuccess: () => void
  categories: any[]
}

export function DocumentUpload({ onUploadSuccess, categories }: DocumentUploadProps) {
  const [files, setFiles] = useState<FileUpload[]>([])
  const [defaultSettings, setDefaultSettings] = useState({
    category: '',
    auto_process: true,
    auto_index: true,
    tags: [] as string[],
    team_access: [] as string[],
  })
  const [newTag, setNewTag] = useState('')
  const [newTeam, setNewTeam] = useState('')

  const uploadMutation = useUploadDocument()

  // Get file type icon
  const getFileTypeIcon = (mimeType: string) => {
    for (const [type, IconComponent] of Object.entries(fileTypeIcons)) {
      if (type !== 'default' && mimeType.startsWith(type)) {
        return IconComponent
      }
    }
    return fileTypeIcons.default
  }

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      errors.forEach((error: any) => {
        if (error.code === 'file-too-large') {
          toast.error(`File "${file.name}" is too large. Maximum size is ${maxFileSize / 1024 / 1024}MB.`)
        } else if (error.code === 'file-invalid-type') {
          toast.error(`File type "${file.type}" is not supported.`)
        }
      })
    })

    // Handle accepted files
    const newFiles: FileUpload[] = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'pending',
      progress: 0,
      metadata: {
        name: file.name.replace(/\.[^/.]+$/, ''), // Remove extension
        description: '',
        category: defaultSettings.category,
        tags: [...defaultSettings.tags],
        team_access: [...defaultSettings.team_access],
        auto_process: defaultSettings.auto_process,
        auto_index: defaultSettings.auto_index
      }
    }))

    setFiles(prev => [...prev, ...newFiles])
  }, [defaultSettings])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxSize: maxFileSize,
    multiple: true
  })

  // Update file metadata
  const updateFileMetadata = (fileId: string, field: string, value: any) => {
    setFiles(prev => prev.map(f => 
      f.id === fileId 
        ? { ...f, metadata: { ...f.metadata, [field]: value } }
        : f
    ))
  }

  // Add tag to file
  const addTagToFile = (fileId: string, tag: string) => {
    if (!tag.trim()) return
    
    updateFileMetadata(fileId, 'tags', [
      ...files.find(f => f.id === fileId)?.metadata.tags || [],
      tag.trim()
    ])
  }

  // Remove tag from file
  const removeTagFromFile = (fileId: string, tagToRemove: string) => {
    const file = files.find(f => f.id === fileId)
    if (!file) return
    
    updateFileMetadata(fileId, 'tags', 
      file.metadata.tags.filter(tag => tag !== tagToRemove)
    )
  }

  // Remove file from upload queue
  const removeFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId))
  }

  // Upload single file
  const uploadFile = async (fileUpload: FileUpload) => {
    console.log('[DocumentUpload] Starting upload for:', fileUpload.file.name, fileUpload.metadata)
    
    try {
      // Update status to uploading
      setFiles(prev => prev.map(f => 
        f.id === fileUpload.id 
          ? { ...f, status: 'uploading', progress: 0 }
          : f
      ))

      // Simulate upload progress (replace with actual progress tracking)
      const progressInterval = setInterval(() => {
        setFiles(prev => prev.map(f => 
          f.id === fileUpload.id && f.progress < 90
            ? { ...f, progress: f.progress + 10 }
            : f
        ))
      }, 200)

      console.log('[DocumentUpload] Calling uploadMutation.mutateAsync...')
      
      // Upload file
      const result = await uploadMutation.mutateAsync({
        file: fileUpload.file,
        metadata: fileUpload.metadata
      })

      console.log('[DocumentUpload] Upload successful! Result:', result)

      clearInterval(progressInterval)

      // Update status to success
      setFiles(prev => prev.map(f => 
        f.id === fileUpload.id 
          ? { ...f, status: 'success', progress: 100 }
          : f
      ))

      toast.success(`✅ ${fileUpload.file.name} uploaded successfully!`)

      // Remove file after 3 seconds
      setTimeout(() => {
        removeFile(fileUpload.id)
      }, 3000)

    } catch (error: any) {
      console.error('[DocumentUpload] Upload failed:', error)
      
      // Update status to error
      setFiles(prev => prev.map(f => 
        f.id === fileUpload.id 
          ? { ...f, status: 'error', error: error.message }
          : f
      ))
      
      toast.error(`❌ Upload failed: ${error.message}`)
    }
  }

  // Upload all files
  const uploadAllFiles = async () => {
    console.log('[DocumentUpload] uploadAllFiles called, files:', files)
    
    const pendingFiles = files.filter(f => f.status === 'pending')
    console.log('[DocumentUpload] pendingFiles:', pendingFiles)
    
    if (pendingFiles.length === 0) {
      console.error('[DocumentUpload] No pending files!')
      toast.error('No files to upload')
      return
    }

    // Validate files
    for (const file of pendingFiles) {
      console.log('[DocumentUpload] Validating file:', file.metadata)
      
      if (!file.metadata.name.trim()) {
        console.error('[DocumentUpload] VALIDATION FAILED: No name for', file.file.name)
        toast.error(`❌ VALIDATION ERROR: Please provide a name for "${file.file.name}"`, { duration: 5000 })
        alert(`VALIDATION ERROR: Please provide a name for "${file.file.name}"`)
        return
      }
      // NOTE: category is not used by backend, removed validation
    }

    // Upload files sequentially (or in batches if needed)
    for (const file of pendingFiles) {
      await uploadFile(file)
    }

    // Call success callback if all uploads succeeded
    if (files.every(f => f.status === 'success' || f.status === 'error')) {
      const successCount = files.filter(f => f.status === 'success').length
      if (successCount > 0) {
        toast.success(`${successCount} file(s) uploaded successfully!`)
        onUploadSuccess()
      }
    }
  }

  // Add tag to default settings
  const addDefaultTag = () => {
    if (!newTag.trim() || defaultSettings.tags.includes(newTag.trim())) return
    
    setDefaultSettings(prev => ({
      ...prev,
      tags: [...prev.tags, newTag.trim()]
    }))
    setNewTag('')
  }

  // Remove tag from default settings
  const removeDefaultTag = (tagToRemove: string) => {
    setDefaultSettings(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }))
  }

  const pendingFiles = files.filter(f => f.status === 'pending')
  const uploadingFiles = files.filter(f => f.status === 'uploading')
  const completedFiles = files.filter(f => f.status === 'success' || f.status === 'error')

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Documents
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive 
                ? 'border-primary bg-primary/5' 
                : 'border-border hover:border-primary/50'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
            {isDragActive ? (
              <p className="text-lg text-primary">Drop the files here...</p>
            ) : (
              <div>
                <p className="text-lg font-medium mb-2">
                  Drag & drop files here, or click to select
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Supports PDF, images, videos, audio, documents, and archives up to {maxFileSize / 1024 / 1024}MB
                </p>
                <Button variant="outline">
                  <Plus className="w-4 h-4 mr-2" />
                  Choose Files
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Default Settings */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Default Settings
            <Badge variant="secondary" className="ml-auto">
              Apply to all uploads
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Default Category</Label>
              <Select
                value={defaultSettings.category}
                onValueChange={(value) => setDefaultSettings(prev => ({ ...prev, category: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No Category</SelectItem>
                  {categories.map(category => (
                    <SelectItem key={category.id} value={category.id}>
                      {category.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Default Tags</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Add tag..."
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      addDefaultTag()
                    }
                  }}
                />
                <Button 
                  type="button" 
                  variant="outline" 
                  size="sm"
                  onClick={addDefaultTag}
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
              {defaultSettings.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {defaultSettings.tags.map(tag => (
                    <Badge 
                      key={tag} 
                      variant="secondary" 
                      className="cursor-pointer"
                      onClick={() => removeDefaultTag(tag)}
                    >
                      {tag} <X className="w-3 h-3 ml-1" />
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label>Team Access</Label>
              <p className="text-xs text-muted-foreground">
                Restrict document visibility to specific teams. Leave empty for all teams.
              </p>
              <div className="flex gap-2">
                <Input
                  placeholder="Add team..."
                  value={newTeam}
                  onChange={(e) => setNewTeam(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      const trimmed = newTeam.trim()
                      if (trimmed && !defaultSettings.team_access.includes(trimmed)) {
                        setDefaultSettings(prev => ({
                          ...prev,
                          team_access: [...prev.team_access, trimmed],
                        }))
                      }
                      setNewTeam('')
                    }
                  }}
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const trimmed = newTeam.trim()
                    if (trimmed && !defaultSettings.team_access.includes(trimmed)) {
                      setDefaultSettings(prev => ({
                        ...prev,
                        team_access: [...prev.team_access, trimmed],
                      }))
                    }
                    setNewTeam('')
                  }}
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
              {defaultSettings.team_access.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {defaultSettings.team_access.map(team => (
                    <Badge
                      key={team}
                      variant="outline"
                      className="cursor-pointer"
                      onClick={() =>
                        setDefaultSettings(prev => ({
                          ...prev,
                          team_access: prev.team_access.filter(t => t !== team),
                        }))
                      }
                    >
                      {team} <X className="w-3 h-3 ml-1" />
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Auto Process</Label>
              <p className="text-sm text-muted-foreground">
                Automatically process documents after upload
              </p>
            </div>
            <Switch
              checked={defaultSettings.auto_process}
              onCheckedChange={(checked) => setDefaultSettings(prev => ({ ...prev, auto_process: checked }))}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Auto Index</Label>
              <p className="text-sm text-muted-foreground">
                Automatically index documents for AI search
              </p>
            </div>
            <Switch
              checked={defaultSettings.auto_index}
              onCheckedChange={(checked) => setDefaultSettings(prev => ({ ...prev, auto_index: checked }))}
            />
          </div>
        </CardContent>
      </Card>

      {/* Upload Queue */}
      {files.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <FolderOpen className="w-5 h-5" />
                Upload Queue
                <Badge variant="secondary">
                  {files.length} file{files.length > 1 ? 's' : ''}
                </Badge>
              </CardTitle>
              
              {pendingFiles.length > 0 && (
                <Button onClick={uploadAllFiles} disabled={uploadMutation.isPending}>
                  {uploadingFiles.length > 0 ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload All
                    </>
                  )}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {files.map(fileUpload => {
                const IconComponent = getFileTypeIcon(fileUpload.file.type)
                
                return (
                  <motion.div
                    key={fileUpload.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="border rounded-lg p-4 space-y-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <IconComponent className="w-8 h-8 text-primary" />
                        <div>
                          <div className="font-medium">{fileUpload.file.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {formatFileSize(fileUpload.file.size)} • {fileUpload.file.type || 'Unknown type'}
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {fileUpload.status === 'success' && (
                          <CheckCircle className="w-5 h-5 text-success" />
                        )}
                        {fileUpload.status === 'error' && (
                          <AlertCircle className="w-5 h-5 text-destructive" />
                        )}
                        {fileUpload.status === 'uploading' && (
                          <Loader2 className="w-5 h-5 animate-spin text-primary" />
                        )}
                        {fileUpload.status === 'pending' && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFile(fileUpload.id)}
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* Upload progress */}
                    {fileUpload.status === 'uploading' && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span>Uploading...</span>
                          <span>{fileUpload.progress}%</span>
                        </div>
                        <Progress value={fileUpload.progress} className="h-2" />
                      </div>
                    )}

                    {/* Error message */}
                    {fileUpload.status === 'error' && fileUpload.error && (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          {fileUpload.error}
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* File metadata (only for pending files) */}
                    {fileUpload.status === 'pending' && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t">
                        <div className="space-y-2">
                          <Label>Document Name</Label>
                          <Input
                            value={fileUpload.metadata.name}
                            onChange={(e) => updateFileMetadata(fileUpload.id, 'name', e.target.value)}
                            placeholder="Enter document name"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Category</Label>
                          <Select
                            value={fileUpload.metadata.category}
                            onValueChange={(value) => updateFileMetadata(fileUpload.id, 'category', value)}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select category" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="none">No Category</SelectItem>
                              {categories.map(category => (
                                <SelectItem key={category.id} value={category.id}>
                                  {category.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="md:col-span-2 space-y-2">
                          <Label>Description (Optional)</Label>
                          <Textarea
                            value={fileUpload.metadata.description}
                            onChange={(e) => updateFileMetadata(fileUpload.id, 'description', e.target.value)}
                            placeholder="Enter document description"
                            rows={2}
                          />
                        </div>

                        <div className="md:col-span-2 space-y-2">
                          <Label>Tags</Label>
                          <div className="flex gap-2">
                            <Input
                              placeholder="Add tag..."
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  e.preventDefault()
                                  const input = e.target as HTMLInputElement
                                  addTagToFile(fileUpload.id, input.value)
                                  input.value = ''
                                }
                              }}
                            />
                          </div>
                          {fileUpload.metadata.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {fileUpload.metadata.tags.map(tag => (
                                <Badge 
                                  key={tag} 
                                  variant="secondary" 
                                  className="cursor-pointer"
                                  onClick={() => removeTagFromFile(fileUpload.id, tag)}
                                >
                                  {tag} <X className="w-3 h-3 ml-1" />
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label>Auto Process</Label>
                            <p className="text-xs text-muted-foreground">
                              Process after upload
                            </p>
                          </div>
                          <Switch
                            checked={fileUpload.metadata.auto_process}
                            onCheckedChange={(checked) => updateFileMetadata(fileUpload.id, 'auto_process', checked)}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label>Auto Index</Label>
                            <p className="text-xs text-muted-foreground">
                              Index for AI search
                            </p>
                          </div>
                          <Switch
                            checked={fileUpload.metadata.auto_index}
                            onCheckedChange={(checked) => updateFileMetadata(fileUpload.id, 'auto_index', checked)}
                          />
                        </div>
                      </div>
                    )}
                  </motion.div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

