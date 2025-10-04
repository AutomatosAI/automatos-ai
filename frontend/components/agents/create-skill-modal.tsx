'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus, 
  X, 
  Save,
  Code,
  Shield,
  Database,
  Zap,
  Brain,
  FileText,
  Settings,
  BarChart
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { toast } from 'react-hot-toast'
import { useCreateSkill } from '@/hooks/use-agent-api'

interface CreateSkillModalProps {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}

const skillCategories = {
  development: { name: 'Development', icon: Code },
  security: { name: 'Security', icon: Shield },
  data: { name: 'Data & Analytics', icon: Database },
  performance: { name: 'Performance', icon: Zap },
  ai: { name: 'AI & ML', icon: Brain },
  documentation: { name: 'Documentation', icon: FileText },
  system: { name: 'System Administration', icon: Settings },
  monitoring: { name: 'Monitoring & Analytics', icon: BarChart }
}

export function CreateSkillModal({ open, onClose, onSuccess }: CreateSkillModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    category: 'development',
    difficulty: 'intermediate',
    version: '1.0.0',
    prerequisites: '',
    learning_objectives: '',
    assessment_criteria: '',
    resources: ''
  })
  const [isSubmitting, setIsSubmitting] = useState(false)

  const createSkillMutation = useCreateSkill()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.name.trim() || !formData.description.trim()) {
      toast.error('Please fill in all required fields')
      return
    }

    setIsSubmitting(true)
    
    try {
      const skillData = {
        name: formData.name.trim(),
        description: formData.description.trim(),
        category: formData.category,
        difficulty: formData.difficulty,
        version: formData.version,
        prerequisites: formData.prerequisites.trim() || null,
        learning_objectives: formData.learning_objectives.trim() || null,
        assessment_criteria: formData.assessment_criteria.trim() || null,
        resources: formData.resources.trim() || null
      }

      await createSkillMutation.mutateAsync(skillData as any)
      onSuccess()
      
      // Reset form
      setFormData({
        name: '',
        description: '',
        category: 'development',
        difficulty: 'intermediate',
        version: '1.0.0',
        prerequisites: '',
        learning_objectives: '',
        assessment_criteria: '',
        resources: ''
      })
    } catch (error) {
      console.error('Failed to create skill:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto bg-gray-900 border border-gray-700">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold text-white flex items-center gap-2">
            <Plus className="w-5 h-5 text-orange-400" />
            Create New Skill
          </DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-white">Basic Information</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="name" className="text-gray-300">Skill Name *</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => handleInputChange('name', e.target.value)}
                  placeholder="e.g., Data Analysis"
                  className="bg-gray-800 border-gray-600 text-white placeholder-gray-400"
                  required
                />
              </div>

              <div>
                <Label htmlFor="category" className="text-gray-300">Category</Label>
                <Select value={formData.category} onValueChange={(value) => handleInputChange('category', value)}>
                  <SelectTrigger className="bg-gray-800 border-gray-600 text-white">
                    <SelectValue placeholder="Select category" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-600">
                    {Object.entries(skillCategories).map(([key, category]) => (
                      <SelectItem key={key} value={key} className="text-white hover:bg-gray-700">
                        <div className="flex items-center gap-2">
                          <category.icon className="w-4 h-4" />
                          {category.name}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label htmlFor="description" className="text-gray-300">Description *</Label>
              <Textarea
                id="description"
                value={formData.description}
                onChange={(e) => handleInputChange('description', e.target.value)}
                placeholder="Describe what this skill enables agents to do..."
                className="bg-gray-800 border-gray-600 text-white placeholder-gray-400 min-h-[100px]"
                required
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="difficulty" className="text-gray-300">Difficulty Level</Label>
                <Select value={formData.difficulty} onValueChange={(value) => handleInputChange('difficulty', value)}>
                  <SelectTrigger className="bg-gray-800 border-gray-600 text-white">
                    <SelectValue placeholder="Select difficulty" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-600">
                    <SelectItem value="beginner" className="text-white hover:bg-gray-700">Beginner</SelectItem>
                    <SelectItem value="intermediate" className="text-white hover:bg-gray-700">Intermediate</SelectItem>
                    <SelectItem value="advanced" className="text-white hover:bg-gray-700">Advanced</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="version" className="text-gray-300">Version</Label>
                <Input
                  id="version"
                  value={formData.version}
                  onChange={(e) => handleInputChange('version', e.target.value)}
                  placeholder="1.0.0"
                  className="bg-gray-800 border-gray-600 text-white placeholder-gray-400"
                />
              </div>
            </div>
          </div>

          {/* Learning Details */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-white">Learning Details</h3>
            
            <div>
              <Label htmlFor="prerequisites" className="text-gray-300">Prerequisites</Label>
              <Textarea
                id="prerequisites"
                value={formData.prerequisites}
                onChange={(e) => handleInputChange('prerequisites', e.target.value)}
                placeholder="What skills or knowledge should agents have before learning this?"
                className="bg-gray-800 border-gray-600 text-white placeholder-gray-400 min-h-[80px]"
              />
            </div>

            <div>
              <Label htmlFor="learning_objectives" className="text-gray-300">Learning Objectives</Label>
              <Textarea
                id="learning_objectives"
                value={formData.learning_objectives}
                onChange={(e) => handleInputChange('learning_objectives', e.target.value)}
                placeholder="What will agents be able to do after mastering this skill?"
                className="bg-gray-800 border-gray-600 text-white placeholder-gray-400 min-h-[80px]"
              />
            </div>

            <div>
              <Label htmlFor="assessment_criteria" className="text-gray-300">Assessment Criteria</Label>
              <Textarea
                id="assessment_criteria"
                value={formData.assessment_criteria}
                onChange={(e) => handleInputChange('assessment_criteria', e.target.value)}
                placeholder="How is mastery of this skill evaluated?"
                className="bg-gray-800 border-gray-600 text-white placeholder-gray-400 min-h-[80px]"
              />
            </div>

            <div>
              <Label htmlFor="resources" className="text-gray-300">Learning Resources</Label>
              <Textarea
                id="resources"
                value={formData.resources}
                onChange={(e) => handleInputChange('resources', e.target.value)}
                placeholder="Links, documentation, or materials for learning this skill"
                className="bg-gray-800 border-gray-600 text-white placeholder-gray-400 min-h-[80px]"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t border-gray-700">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isSubmitting}
              className="border-gray-600 text-gray-300 hover:bg-gray-800"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isSubmitting}
              className="bg-orange-500 hover:bg-orange-600 text-white"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Creating...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Create Skill
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
