
'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus, 
  Search, 
  Star, 
  Edit, 
  Trash2, 
  Code, 
  Shield, 
  Database, 
  Zap,
  Brain,
  FileText,
  Settings,
  BarChart,
  Filter,
  Tag
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'react-hot-toast'

// API hooks
import { useSkills, useAgentSkills, useAddSkillToAgent, useRemoveSkillFromAgent } from '@/hooks/use-agent-api'

import { CreateSkillModal } from "./create-skill-modal"
const skillCategories = {
  development: {
    name: 'Development',
    icon: Code,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10'
  },
  security: {
    name: 'Security',
    icon: Shield,
    color: 'text-red-400',
    bgColor: 'bg-red-500/10'
  },
  data: {
    name: 'Data & Analytics',
    icon: Database,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10'
  },
  performance: {
    name: 'Performance',
    icon: Zap,
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/10'
  },
  ai: {
    name: 'AI & ML',
    icon: Brain,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10'
  },
  documentation: {
    name: 'Documentation',
    icon: FileText,
    color: 'text-indigo-400',
    bgColor: 'bg-indigo-500/10'
  },
  system: {
    name: 'System Administration',
    icon: Settings,
    color: 'text-gray-400',
    bgColor: 'bg-gray-500/10'
  },
  monitoring: {
    name: 'Monitoring & Analytics',
    icon: BarChart,
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10'
  }
}

interface RealAgentSkillsProps {
  agents: any[]
  selectedAgentId: string | null
  onAgentSelect: (agentId: string | null) => void
}

export function RealAgentSkills({ agents, selectedAgentId, onAgentSelect }: RealAgentSkillsProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [difficultyFilter, setDifficultyFilter] = useState('all')
  const [activeTab, setActiveTab] = useState('all-skills')
  const [showCreateModal, setShowCreateModal] = useState(false)

  // Fetch skills data
  const { data: allSkills = [], isLoading: skillsLoading } = useSkills()
  const { data: agentSkills = [], isLoading: agentSkillsLoading } = useAgentSkills(selectedAgentId)

  // API mutations
  const addSkillMutation = useAddSkillToAgent()
  const removeSkillMutation = useRemoveSkillFromAgent()

  // Get selected agent
  const selectedAgent = selectedAgentId ? agents.find(a => a.id === selectedAgentId) : null

  // Filter skills
  const filteredSkills = useMemo(() => {
    return allSkills.filter(skill => {
      const matchesSearch = !searchTerm || 
        skill.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        skill.description?.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesCategory = categoryFilter === 'all' || skill.category === categoryFilter
      const matchesDifficulty = difficultyFilter === 'all' || skill.difficulty === difficultyFilter
      
      return matchesSearch && matchesCategory && matchesDifficulty
    })
  }, [allSkills, searchTerm, categoryFilter, difficultyFilter])

  // Group skills by category
  const skillsByCategory = useMemo(() => {
    const grouped: Record<string, any[]> = {}
    filteredSkills.forEach(skill => {
      const category = skill.category || 'other'
      if (!grouped[category]) {
        grouped[category] = []
      }
      grouped[category].push(skill)
    })
    return grouped
  }, [filteredSkills])

  // Handle adding skill to agent
  const handleAddSkill = async (skillId: string) => {
    if (!selectedAgentId) {
      toast.error('Please select an agent first')
      return
    }

    try {
      await addSkillMutation.mutateAsync({ agentId: selectedAgentId, skillId })
      toast.success('Skill added to agent successfully')
    } catch (error) {
      toast.error('Failed to add skill to agent')
    }
  }

  // Handle removing skill from agent
  const handleRemoveSkill = async (skillId: string) => {
    if (!selectedAgentId) {
      return
    }

    try {
      await removeSkillMutation.mutateAsync({ agentId: selectedAgentId, skillId })
      toast.success('Skill removed from agent successfully')
    } catch (error) {
      toast.error('Failed to remove skill from agent')
    }
  }

  // Check if agent has skill
  const agentHasSkill = (skillId: string) => {
    return Array.isArray(agentSkills) ? agentSkills.some : false(skill => skill.id === skillId)
  }

  if (skillsLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="glass-card">
              <CardHeader>
                <Skeleton className="h-6 w-32" />
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row gap-4 lg:items-center lg:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Skills Management</h2>
          <p className="text-muted-foreground">
            Manage agent skills and capabilities across your system
          </p>
        </div>

        <Button
          onClick={() => setShowCreateModal(true)}
          className="bg-orange-500 hover:bg-orange-600"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Skill
        </Button>

        {/* Agent Selector */}
        <div className="flex items-center gap-4">
          <div className="min-w-[200px]">
            <Select value={selectedAgentId || ''} onValueChange={(value) => onAgentSelect(value || null)}>
              <SelectTrigger>
                <SelectValue placeholder="Select an agent" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Agents</SelectItem>
                {agents.map(agent => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Selected Agent Info */}
      {selectedAgent && (
        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-white text-xl">
                🤖
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold">{selectedAgent.name}</h3>
                <p className="text-sm text-muted-foreground">
                  {selectedAgent.agent_type?.replace('_', ' ')} • {Array.isArray(agentSkills) ? agentSkills.length : 0} skills assigned
                </p>
              </div>
              <Badge variant="outline" className="capitalize">
                {selectedAgent.status}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="all-skills">All Skills</TabsTrigger>
          <TabsTrigger value="agent-skills" disabled={!selectedAgentId}>
            Agent Skills {selectedAgentId && `(${Array.isArray(agentSkills) ? agentSkills.length : 0})`}
          </TabsTrigger>
          <TabsTrigger value="skill-categories">Categories</TabsTrigger>
        </TabsList>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search skills..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          
          <Select value={categoryFilter} onValueChange={setCategoryFilter}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All Categories" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              {Object.entries(skillCategories).map(([key, category]) => (
                <SelectItem key={key} value={key}>
                  {category.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Select value={difficultyFilter} onValueChange={setDifficultyFilter}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All Difficulties" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Difficulties</SelectItem>
              <SelectItem value="beginner">Beginner</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* All Skills Tab */}
        <TabsContent value="all-skills" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredSkills.map((skill, index) => {
              const category = skillCategories[skill.category] || skillCategories.development
              const hasSkill = agentHasSkill(skill.id)
              
              return (
                <motion.div
                  key={skill.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <Card className={`glass-card transition-all duration-200 hover:shadow-lg ${
                    hasSkill ? 'border-green-500/30 bg-green-500/5' : ''
                  }`}>
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${category.bgColor}`}>
                            <category.icon className={`w-5 h-5 ${category.color}`} />
                          </div>
                          <div>
                            <CardTitle className="text-lg">{skill.name}</CardTitle>
                            <Badge variant="secondary" className="text-xs mt-1">
                              {skill.difficulty}
                            </Badge>
                          </div>
                        </div>
                        
                        {selectedAgentId && (
                          <Button
                            size="sm"
                            variant={hasSkill ? "destructive" : "default"}
                            onClick={() => hasSkill ? handleRemoveSkill(skill.id) : handleAddSkill(skill.id)}
                            disabled={addSkillMutation.isPending || removeSkillMutation.isPending}
                          >
                            {hasSkill ? (
                              <>
                                <Trash2 className="w-4 h-4 mr-1" />
                                Remove
                              </>
                            ) : (
                              <>
                                <Plus className="w-4 h-4 mr-1" />
                                Add
                              </>
                            )}
                          </Button>
                        )}
                      </div>
                    </CardHeader>
                    
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-3">
                        {skill.description}
                      </p>
                      
                      <div className="flex items-center justify-between">
                        <Badge variant="outline" className={category.color}>
                          {category.name}
                        </Badge>
                        
                        {hasSkill && (
                          <div className="flex items-center gap-1 text-green-600">
                            <Star className="w-4 h-4 fill-current" />
                            <span className="text-xs">Assigned</span>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )
            })}
          </div>
        </TabsContent>

        {/* Agent Skills Tab */}
        <TabsContent value="agent-skills" className="space-y-6">
          {selectedAgentId ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.isArray(agentSkills) ? agentSkills.map : [].map((skill, index) => {
                const category = skillCategories[skill.category] || skillCategories.development
                
                return (
                  <motion.div
                    key={skill.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <Card className="glass-card border-green-500/30 bg-green-500/5">
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg ${category.bgColor}`}>
                              <category.icon className={`w-5 h-5 ${category.color}`} />
                            </div>
                            <div>
                              <CardTitle className="text-lg">{skill.name}</CardTitle>
                              <Badge variant="secondary" className="text-xs mt-1">
                                {skill.difficulty}
                              </Badge>
                            </div>
                          </div>
                          
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => handleRemoveSkill(skill.id)}
                            disabled={removeSkillMutation.isPending}
                          >
                            <Trash2 className="w-4 h-4 mr-1" />
                            Remove
                          </Button>
                        </div>
                      </CardHeader>
                      
                      <CardContent>
                        <p className="text-sm text-muted-foreground mb-3">
                          {skill.description}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <Badge variant="outline" className={category.color}>
                            {category.name}
                          </Badge>
                          
                          <div className="flex items-center gap-1 text-green-600">
                            <Star className="w-4 h-4 fill-current" />
                            <span className="text-xs">Active</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          ) : (
            <Card className="glass-card">
              <CardContent className="p-12 text-center">
                <Tag className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">Select an Agent</h3>
                <p className="text-muted-foreground">
                  Choose an agent to view and manage their assigned skills
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Categories Tab */}
        <TabsContent value="skill-categories" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(skillsByCategory).map(([categoryKey, skills]) => {
              const category = skillCategories[categoryKey] || skillCategories.development
              
              return (
                <Card key={categoryKey} className="glass-card">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className={`p-3 rounded-lg ${category.bgColor}`}>
                        <category.icon className={`w-6 h-6 ${category.color}`} />
                      </div>
                      <div>
                        <CardTitle className="text-lg">{category.name}</CardTitle>
                        <p className="text-sm text-muted-foreground">
                          {skills.length} skills
                        </p>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {skills.slice(0, 3).map(skill => (
                        <div key={skill.id} className="text-sm">
                          {skill.name}
                        </div>
                      ))}
                      {skills.length > 3 && (
                        <div className="text-xs text-muted-foreground">
                          +{skills.length - 3} more skills
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>
      </Tabs>
      {/* Create Skill Modal */}
      <CreateSkillModal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={() => {
          setShowCreateModal(false)
          // Skills will auto-refresh via React Query
        }}
      />
    </div>
  )
}

