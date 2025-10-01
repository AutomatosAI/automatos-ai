'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus,
  GitBranch,
  Users,
  CheckCircle,
  ChevronRight,
  Edit,
  Trash2,
  AlertTriangle,
  Search,
  Filter,
  Eye,
  Clock,
  Star
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  useWorkflowTemplates, 
  useCreateTemplate, 
  useUpdateTemplate, 
  useDeleteTemplate,
  useRecordTemplateUsage 
} from '@/hooks/use-template-api'

interface TemplatesTabProps {
  onUseTemplate: (templateId: string) => void
  onOpenCreateModal?: () => void
}

export function TemplatesTab({ onUseTemplate, onOpenCreateModal }: TemplatesTabProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<any>(null)
  const [templateForm, setTemplateForm] = useState<any>({
    template_id: '',
    name: '',
    description: '',
    category: 'Development',
    difficulty: 'intermediate',
    tags: [],
    icon: '📋',
    template_definition: {
      steps: [],
      agents: [],
      config: {}
    },
    recommended_agents: [],
    estimated_time: '',
    is_public: true,
    is_featured: false
  })

  // Fetch templates with filtering
  const { data: templatesData, isLoading, error, refetch } = useWorkflowTemplates({
    category: selectedCategory !== 'all' ? selectedCategory : undefined,
    difficulty: selectedDifficulty !== 'all' ? selectedDifficulty : undefined,
    search: searchTerm || undefined,
    limit: 100
  })

  const createMutation = useCreateTemplate()
  const updateMutation = useUpdateTemplate()
  const deleteMutation = useDeleteTemplate()
  const recordUsageMutation = useRecordTemplateUsage()

  const templates = templatesData?.items || []

  const handleCreateTemplate = async () => {
    try {
      await createMutation.mutateAsync({
        ...templateForm,
        created_by: 'user'
      })
      setShowCreateModal(false)
      resetForm()
      refetch()
    } catch (error) {
      console.error('Error creating template:', error)
    }
  }

  const handleUpdateTemplate = async () => {
    if (!selectedTemplate) return
    try {
      await updateMutation.mutateAsync({
        templateId: selectedTemplate.template_id,
        templateData: templateForm
      })
      setShowEditModal(false)
      setSelectedTemplate(null)
      resetForm()
      refetch()
    } catch (error) {
      console.error('Error updating template:', error)
    }
  }

  const handleDeleteTemplate = async () => {
    if (!selectedTemplate) return
    try {
      await deleteMutation.mutateAsync(selectedTemplate.template_id)
      setShowDeleteDialog(false)
      setSelectedTemplate(null)
      refetch()
    } catch (error) {
      console.error('Error deleting template:', error)
    }
  }

  const handleUseTemplate = async (template: any) => {
    try {
      // Record usage
      await recordUsageMutation.mutateAsync(template.template_id)
      // Notify parent component
      onUseTemplate(template.template_id)
      if (onOpenCreateModal) {
        onOpenCreateModal()
      }
    } catch (error) {
      console.error('Error recording template usage:', error)
    }
  }

  const handleViewClick = (template: any) => {
    setSelectedTemplate(template)
    setShowViewModal(true)
  }

  const handleEditClick = (template: any) => {
    setSelectedTemplate(template)
    setTemplateForm({
      template_id: template.template_id,
      name: template.name,
      description: template.description,
      category: template.category,
      difficulty: template.difficulty,
      tags: template.tags || [],
      icon: template.icon || '📋',
      template_definition: template.template_definition || { steps: [], agents: [], config: {} },
      recommended_agents: template.recommended_agents || [],
      estimated_time: template.estimated_time || '',
      is_public: template.is_public,
      is_featured: template.is_featured
    })
    setShowEditModal(true)
  }

  const handleDeleteClick = (template: any) => {
    setSelectedTemplate(template)
    setShowDeleteDialog(true)
  }

  const resetForm = () => {
    setTemplateForm({
      template_id: '',
      name: '',
      description: '',
      category: 'Development',
      difficulty: 'intermediate',
      tags: [],
      icon: '📋',
      template_definition: {
        steps: [],
        agents: [],
        config: {}
      },
      recommended_agents: [],
      estimated_time: '',
      is_public: true,
      is_featured: false
    })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading templates...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400">Error loading templates</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex-1 flex gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search templates..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 bg-secondary/50 border-secondary"
            />
          </div>
          
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-[180px] bg-secondary/50 border-secondary">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              <SelectItem value="Support">Support</SelectItem>
              <SelectItem value="Data Processing">Data Processing</SelectItem>
              <SelectItem value="Content">Content</SelectItem>
              <SelectItem value="Security">Security</SelectItem>
              <SelectItem value="Development">Development</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedDifficulty} onValueChange={setSelectedDifficulty}>
            <SelectTrigger className="w-[180px] bg-secondary/50 border-secondary">
              <SelectValue placeholder="Difficulty" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="beginner">Beginner</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Button 
          onClick={() => setShowCreateModal(true)}
          className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Template
        </Button>
      </div>

      {/* Templates Grid */}
      {templates.length === 0 ? (
        <div className="text-center py-12">
          <GitBranch className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No templates found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {templates.map((template: any, index: number) => (
            <motion.div
              key={template.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="glass-card card-glow hover:border-primary/30 transition-all duration-300 h-full flex flex-col">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-2xl">{template.icon || '📋'}</span>
                        <CardTitle className="text-lg">{template.name}</CardTitle>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {template.description}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="outline" className="text-xs">
                          {template.category}
                        </Badge>
                        <Badge variant="outline" className="text-xs capitalize">
                          {template.difficulty}
                        </Badge>
                        {template.is_system && (
                          <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                            System
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-between space-y-4">
                  {/* Template Stats */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Used {template.use_count} times</span>
                      {template.estimated_time && <span>{template.estimated_time}</span>}
                    </div>
                    
                    {/* Recommended Agents */}
                    {template.recommended_agents && template.recommended_agents.length > 0 && (
                      <div>
                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                          <Users className="w-3 h-3 mr-1" />
                          Recommended Agents
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {template.recommended_agents.slice(0, 3).map((agent: string, idx: number) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {agent}
                            </Badge>
                          ))}
                          {template.recommended_agents.length > 3 && (
                            <span className="text-xs text-muted-foreground">
                              +{template.recommended_agents.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Button 
                      className="flex-1 bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
                      size="sm"
                      onClick={() => handleUseTemplate(template)}
                    >
                      <Plus className="w-3 h-3 mr-2" />
                      Use
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleViewClick(template)}
                    >
                      <Eye className="w-3 h-3" />
                    </Button>
                    {!template.is_system && (
                      <>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleEditClick(template)}
                        >
                          <Edit className="w-3 h-3" />
                        </Button>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleDeleteClick(template)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {/* Create Template Modal */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Workflow Template</DialogTitle>
            <DialogDescription>
              Create a reusable template that others can use to quickly start workflows
            </DialogDescription>
          </DialogHeader>
          
          <TemplateForm
            form={templateForm}
            onChange={setTemplateForm}
            onSubmit={handleCreateTemplate}
            onCancel={() => {
              setShowCreateModal(false)
              resetForm()
            }}
            isLoading={createMutation.isPending}
            submitLabel="Create Template"
          />
        </DialogContent>
      </Dialog>

      {/* View Template Modal */}
      <Dialog open={showViewModal} onOpenChange={setShowViewModal}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-3xl">{selectedTemplate?.icon || '📋'}</span>
                <div>
                  <DialogTitle className="text-2xl">{selectedTemplate?.name}</DialogTitle>
                  <p className="text-sm text-muted-foreground mt-1">{selectedTemplate?.template_id}</p>
                </div>
              </div>
              {!selectedTemplate?.is_system && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => {
                    setShowViewModal(false)
                    handleEditClick(selectedTemplate)
                  }}
                >
                  <Edit className="w-3 h-3 mr-2" />
                  Edit
                </Button>
              )}
            </div>
          </DialogHeader>
          
          {selectedTemplate && (
            <div className="space-y-6">
              {/* Badges */}
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="text-xs">{selectedTemplate.category}</Badge>
                <Badge variant="outline" className="text-xs capitalize">{selectedTemplate.difficulty}</Badge>
                {selectedTemplate.is_system && (
                  <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                    System Template
                  </Badge>
                )}
                {selectedTemplate.is_featured && (
                  <Badge variant="outline" className="text-xs bg-yellow-500/10 text-yellow-400 border-yellow-500/20">
                    <Star className="w-3 h-3 mr-1" />
                    Featured
                  </Badge>
                )}
              </div>

              {/* Description */}
              <div>
                <h3 className="text-sm font-semibold mb-2">Description</h3>
                <p className="text-sm text-muted-foreground">{selectedTemplate.description}</p>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-secondary/30 rounded-lg">
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Used</div>
                  <div className="text-lg font-semibold">{selectedTemplate.use_count} times</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Success Rate</div>
                  <div className="text-lg font-semibold">{selectedTemplate.success_rate || 0}%</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">
                    <Clock className="w-3 h-3 inline mr-1" />
                    Duration
                  </div>
                  <div className="text-lg font-semibold">{selectedTemplate.estimated_time || 'N/A'}</div>
                </div>
              </div>

              {/* Recommended Agents */}
              {selectedTemplate.recommended_agents && selectedTemplate.recommended_agents.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold mb-2 flex items-center">
                    <Users className="w-4 h-4 mr-2" />
                    Recommended Agents ({selectedTemplate.recommended_agents.length})
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedTemplate.recommended_agents.map((agent: string, idx: number) => (
                      <Badge key={idx} variant="outline">{agent}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Tags */}
              {selectedTemplate.tags && selectedTemplate.tags.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold mb-2">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedTemplate.tags.map((tag: string, idx: number) => (
                      <Badge key={idx} variant="secondary" className="text-xs">#{tag}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Template Definition */}
              {selectedTemplate.template_definition && (
                <div>
                  <h3 className="text-sm font-semibold mb-2 flex items-center">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Workflow Steps
                  </h3>
                  {selectedTemplate.template_definition.steps && selectedTemplate.template_definition.steps.length > 0 ? (
                    <div className="space-y-2">
                      {selectedTemplate.template_definition.steps.map((step: any, idx: number) => (
                        <div key={idx} className="flex items-start p-3 bg-secondary/30 rounded-lg">
                          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-semibold mr-3 flex-shrink-0">
                            {idx + 1}
                          </div>
                          <div className="flex-1">
                            <div className="font-medium text-sm">{step.name || step}</div>
                            {step.type && <div className="text-xs text-muted-foreground mt-1">Type: {step.type}</div>}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No steps defined</p>
                  )}
                </div>
              )}

              {/* Metadata */}
              <div className="pt-4 border-t border-secondary text-xs text-muted-foreground space-y-1">
                <div>Created by: {selectedTemplate.created_by}</div>
                <div>Version: {selectedTemplate.version}</div>
                {selectedTemplate.created_at && (
                  <div>Created: {new Date(selectedTemplate.created_at).toLocaleDateString()}</div>
                )}
                {selectedTemplate.last_used_at && (
                  <div>Last used: {new Date(selectedTemplate.last_used_at).toLocaleDateString()}</div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4 border-t border-secondary">
                <Button 
                  className="flex-1 bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
                  onClick={() => {
                    setShowViewModal(false)
                    handleUseTemplate(selectedTemplate)
                  }}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Use This Template
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => setShowViewModal(false)}
                >
                  Close
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Edit Template Modal */}
      <Dialog open={showEditModal} onOpenChange={setShowEditModal}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Template</DialogTitle>
            <DialogDescription>
              Update template information and configuration
            </DialogDescription>
          </DialogHeader>
          
          <TemplateForm
            form={templateForm}
            onChange={setTemplateForm}
            onSubmit={handleUpdateTemplate}
            onCancel={() => {
              setShowEditModal(false)
              setSelectedTemplate(null)
              resetForm()
            }}
            isLoading={updateMutation.isPending}
            submitLabel="Update Template"
            isEdit
          />
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Template?</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{selectedTemplate?.name}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          
          <div className="flex justify-end gap-4 mt-6">
            <Button 
              variant="outline" 
              onClick={() => {
                setShowDeleteDialog(false)
                setSelectedTemplate(null)
              }}
            >
              Cancel
            </Button>
            <Button 
              variant="destructive"
              onClick={handleDeleteTemplate}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? 'Deleting...' : 'Delete Template'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// Template Form Component
interface TemplateFormProps {
  form: any
  onChange: (form: any) => void
  onSubmit: () => void
  onCancel: () => void
  isLoading: boolean
  submitLabel: string
  isEdit?: boolean
}

function TemplateForm({ form, onChange, onSubmit, onCancel, isLoading, submitLabel, isEdit }: TemplateFormProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="template_id">Template ID *</Label>
          <Input
            id="template_id"
            value={form.template_id}
            onChange={(e) => onChange({ ...form, template_id: e.target.value })}
            placeholder="my-custom-template"
            disabled={isEdit}
            className="bg-secondary/50 border-secondary"
          />
          <p className="text-xs text-muted-foreground mt-1">
            Unique identifier (lowercase, hyphens only)
          </p>
        </div>
        
        <div>
          <Label htmlFor="icon">Icon</Label>
          <Input
            id="icon"
            value={form.icon}
            onChange={(e) => onChange({ ...form, icon: e.target.value })}
            placeholder="📋"
            className="bg-secondary/50 border-secondary"
          />
        </div>
      </div>

      <div>
        <Label htmlFor="name">Template Name *</Label>
        <Input
          id="name"
          value={form.name}
          onChange={(e) => onChange({ ...form, name: e.target.value })}
          placeholder="My Workflow Template"
          className="bg-secondary/50 border-secondary"
        />
      </div>

      <div>
        <Label htmlFor="description">Description *</Label>
        <Textarea
          id="description"
          value={form.description}
          onChange={(e) => onChange({ ...form, description: e.target.value })}
          placeholder="Describe what this template does..."
          className="bg-secondary/50 border-secondary min-h-[80px]"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <Label htmlFor="category">Category *</Label>
          <Select 
            value={form.category}
            onValueChange={(value) => onChange({ ...form, category: value })}
          >
            <SelectTrigger className="bg-secondary/50 border-secondary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Support">Support</SelectItem>
              <SelectItem value="Data Processing">Data Processing</SelectItem>
              <SelectItem value="Content">Content</SelectItem>
              <SelectItem value="Security">Security</SelectItem>
              <SelectItem value="Development">Development</SelectItem>
              <SelectItem value="Marketing">Marketing</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="difficulty">Difficulty *</Label>
          <Select 
            value={form.difficulty}
            onValueChange={(value) => onChange({ ...form, difficulty: value })}
          >
            <SelectTrigger className="bg-secondary/50 border-secondary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="beginner">Beginner</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="estimated_time">Estimated Time</Label>
          <Input
            id="estimated_time"
            value={form.estimated_time}
            onChange={(e) => onChange({ ...form, estimated_time: e.target.value })}
            placeholder="5-10 minutes"
            className="bg-secondary/50 border-secondary"
          />
        </div>
      </div>

      <div className="flex justify-end gap-4 pt-4 border-t border-secondary">
        <Button variant="outline" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>
        <Button 
          onClick={onSubmit} 
          disabled={isLoading || !form.template_id || !form.name || !form.description}
          className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
        >
          {isLoading ? 'Saving...' : submitLabel}
        </Button>
      </div>
    </div>
  )
}

