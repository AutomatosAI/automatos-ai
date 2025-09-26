'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Zap, 
  Code, 
  Brain, 
  MessageSquare, 
  Settings,
  Plus,
  Check
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { toast } from 'react-hot-toast'

// API hooks
import { useCreateSkill } from '@/hooks/use-agent-api'

interface CreateSkillModalProps {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}

const skillTypes = [
  {
    type: 'technical',
    name: 'Technical',
    description: 'Programming, system administration, and technical implementation',
    icon: Code,
    color: 'text-blue-400'
  },
  {
    type: 'cognitive',
    name: 'Cognitive',
    description: 'Analysis, problem-solving, and decision-making capabilities',
    icon: Brain,
    color: 'text-purple-400'
  },
  {
    type: 'communication',
    name: 'Communication',
    description: 'Interaction, messaging, and information exchange',
    icon: MessageSquare,
    color: 'text-green-400'
  }
]

const skillCategories = [
  {
    category: 'development',
    name: 'Development',
    description: 'Software development and programming'
  },
  {
    category: 'analysis',
    name: 'Analysis',
    description: 'Data analysis and insights'
  },
  {
    category: 'communication',
    name: 'Communication',
    description: 'Messaging and interaction'
  },
  {
    category: 'technical',
    name: 'Technical',
    description: 'Technical operations and systems'
  },
  {
    category: 'management',
    name: 'Management',
    description: 'Project and resource management'
  }
]

export function CreateSkillModal({ open, onClose, onSuccess }: CreateSkillModalProps) {
  const [step, setStep] = useState(1)
  const [skillData, setSkillData] = useState({
    name: '',
    description: '',
    skill_type: '',
    category: '',
    implementation: '',
    parameters: {}
  })
  const [isSubmitting, setIsSubmitting] = useState(false)

  const createSkillMutation = useCreateSkill()

  const handleSubmit = async () => {
    if (!skillData.name || !skillData.skill_type || !skillData.category) {
      toast.error('Please fill in all required fields')
      return
    }

    setIsSubmitting(true)
    try {
      // Format the skill data correctly
      const formattedData = {
        name: skillData.name,
        description: skillData.description,
        skill_type: skillData.skill_type,
        category: skillData.category,
        difficulty: "intermediate", // Default difficulty level
        is_active: true,
        implementation: skillData.implementation || "",
        parameters: skillData.parameters || {}
      }
      
      await createSkillMutation.mutateAsync(formattedData)
      toast.success(`Skill ${skillData.name} created successfully`)
      onSuccess()
      resetForm()
      handleClose()
    } catch (error: any) {
      console.error('Error creating skill:', error)
      toast.error(`Failed to create skill: ${error?.message || 'Unknown error'}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  const resetForm = () => {
    setStep(1)
    setSkillData({
      name: '',
      description: '',
      skill_type: '',
      category: '',
      implementation: '',
      parameters: {}
    })
  }

  const handleClose = () => {
    resetForm()
    onClose()
  }

  if (!open) return null

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={handleClose}
        />

        {/* Modal */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="relative w-full max-w-2xl mx-4 bg-card border border-border rounded-xl shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-orange-500/10 rounded-lg">
                <Zap className="w-6 h-6 text-orange-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold">Create New Skill</h2>
                <p className="text-sm text-muted-foreground">
                  Add a new skill to the agent capabilities library
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClose}
              className="h-8 w-8 p-0"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Content */}
          <div className="p-6">
            {step === 1 && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Basic Information</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="skill-name">Skill Name *</Label>
                      <Input
                        id="skill-name"
                        placeholder="e.g., Python Programming, Data Analysis"
                        value={skillData.name}
                        onChange={(e) => setSkillData({ ...skillData, name: e.target.value })}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label htmlFor="skill-description">Description</Label>
                      <Textarea
                        id="skill-description"
                        placeholder="Describe what this skill does and its capabilities..."
                        value={skillData.description}
                        onChange={(e) => setSkillData({ ...skillData, description: e.target.value })}
                        className="mt-1"
                        rows={3}
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Skill Type *</Label>
                        <Select
                          value={skillData.skill_type}
                          onValueChange={(value) => setSkillData({ ...skillData, skill_type: value })}
                        >
                          <SelectTrigger className="mt-1">
                            <SelectValue placeholder="Select skill type" />
                          </SelectTrigger>
                          <SelectContent>
                            {skillTypes.map((type) => (
                              <SelectItem key={type.type} value={type.type}>
                                <div className="flex items-center space-x-2">
                                  <type.icon className={`w-4 h-4 ${type.color}`} />
                                  <span>{type.name}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label>Category *</Label>
                        <Select
                          value={skillData.category}
                          onValueChange={(value) => setSkillData({ ...skillData, category: value })}
                        >
                          <SelectTrigger className="mt-1">
                            <SelectValue placeholder="Select category" />
                          </SelectTrigger>
                          <SelectContent>
                            {skillCategories.map((cat) => (
                              <SelectItem key={cat.category} value={cat.category}>
                                {cat.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-between">
                  <Button variant="outline" onClick={handleClose}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={() => setStep(2)}
                    disabled={!skillData.name || !skillData.skill_type || !skillData.category}
                  >
                    Next: Implementation
                  </Button>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Implementation Details</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="implementation">Implementation</Label>
                      <Textarea
                        id="implementation"
                        placeholder="Code, configuration, or implementation details (optional)..."
                        value={skillData.implementation}
                        onChange={(e) => setSkillData({ ...skillData, implementation: e.target.value })}
                        className="mt-1"
                        rows={4}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Optional: Add code, configuration, or implementation details
                      </p>
                    </div>
                  </div>
                </div>

                {/* Preview */}
                <div>
                  <h4 className="text-sm font-medium mb-3">Preview</h4>
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <h5 className="font-medium">{skillData.name || 'Skill Name'}</h5>
                            <Badge variant="secondary" className="text-xs">
                              {skillData.skill_type || 'Type'}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {skillData.category || 'Category'}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {skillData.description || 'Skill description will appear here...'}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <div className="flex justify-between">
                  <Button variant="outline" onClick={() => setStep(1)}>
                    Back
                  </Button>
                  <Button 
                    onClick={handleSubmit}
                    disabled={isSubmitting || !skillData.name}
                    className="min-w-[120px]"
                  >
                    {isSubmitting ? (
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span>Creating...</span>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <Plus className="w-4 h-4" />
                        <span>Create Skill</span>
                      </div>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
