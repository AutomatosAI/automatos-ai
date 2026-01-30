
'use client'

import { useState, useMemo, useEffect } from 'react'
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
  Tag,
  MoreVertical,
  Eye
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
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
import { 
  useAgentSkills, 
  useAddSkillToAgent, 
  useRemoveSkillFromAgent,
  useDeleteSkill,
  useUpdateSkill
} from '@/hooks/use-agent-api'
import { useSkillsApi } from '@/hooks/use-skills-api'
import { apiClient } from '@/lib/api-client'

import { CreateSkillModal } from "./create-skill-modal"
import { SkillConfigurationModal } from "./skill-configuration-modal"
// PRD-22 Skills UI components
import { ImportGitModal } from './skills/import-git-modal'
import { SkillSourceList } from './skills/skill-source-list'
import { SkillBrowser } from './skills/skill-browser'
type SkillCategory = {
  name: string
  icon: any
  color: string
  bgColor: string
}

const skillCategories: Record<string, SkillCategory> = {
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

interface AgentSkillsProps {
  agents: any[]
  selectedAgentId: string | null
  onAgentSelect: (agentId: string | null) => void
}

export function AgentSkills({ agents, selectedAgentId, onAgentSelect }: AgentSkillsProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [difficultyFilter, setDifficultyFilter] = useState('all')
  const [activeTab, setActiveTab] = useState('all-skills')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showConfigureModal, setShowConfigureModal] = useState(false)
  const [selectedSkillForConfig, setSelectedSkillForConfig] = useState<any>(null)
  // PRD-22: Git import modal
  const [showImportGit, setShowImportGit] = useState(false)

  // PRD-22: Fetch skills via Skills API
  const { listSkills, loading: skillsLoading } = useSkillsApi()
  const [allSkills, setAllSkills] = useState<any[]>([])
  const [availableCategories, setAvailableCategories] = useState<string[]>([])

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const data = await listSkills({ limit: 200 })
        const items = Array.isArray(data)
          ? data
          : (data?.items || data?.skills || [])
        if (!mounted) return
        let results = items
        // PRD-22: Direct API fallback with supported limit only if empty
        if (!results || results.length === 0) {
          try {
            const base = apiClient.getBaseUrl()
            const url = base ? `${base}/api/v1/skills?limit=200` : '/api/v1/skills?limit=200'
            const res = await fetch(url)
            if (res.ok) {
              const json = await res.json()
              const maybe = Array.isArray(json) ? json : (json?.items || json?.skills || [])
              results = maybe || []
            }
          } catch (_) { /* ignore */ }
        }
        if (!mounted) return
        setAllSkills(results)
        const cats = Array.from(new Set(
          (results || []).map((s: any) => (s.category || 'other')).filter(Boolean)
        ))
        setAvailableCategories(cats)
      } catch (_) {
        if (!mounted) return
        setAllSkills([])
        setAvailableCategories([])
      }
    })()
    return () => { mounted = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Run once on mount
  const { data: agentSkills = [], isLoading: agentSkillsLoading } = useAgentSkills(selectedAgentId)

  // API mutations
  const addSkillMutation = useAddSkillToAgent()
  const removeSkillMutation = useRemoveSkillFromAgent()
  const updateSkillMutation = useUpdateSkill()
  const deleteSkillMutation = useDeleteSkill()

  // Get selected agent
  const selectedAgent = selectedAgentId ? agents.find(a => a.id === selectedAgentId) : null

  // Filter skills
  const filteredSkills = useMemo(() => {
    return allSkills.filter(skill => {
      const matchesSearch = !searchTerm || 
        skill.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        skill.description?.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesCategory = categoryFilter === 'all' || skill.category === categoryFilter
      const matchesDifficulty = difficultyFilter === 'all' || skill.skill_type === difficultyFilter
      
      return matchesSearch && matchesCategory && matchesDifficulty
    })
  }, [allSkills, searchTerm, categoryFilter, difficultyFilter])

  // Group skills by category
  const skillsByCategory = useMemo(() => {
    const grouped: Record<string, any[]> = {}
    filteredSkills.forEach(skill => {
      const category = skill.category || 'other'
      if (!grouped[category]) grouped[category] = []
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
  
  // Handle viewing skill details
  const handleViewSkillDetails = (skill: any) => {
    // For now, just show a toast with basic information
    toast.success(`Viewing details for skill: ${skill.name}`)
    // In the future, this could open a modal with full skill details
  }
  
  // Handle skill configuration/editing
  const handleConfigureSkill = (skill: any) => {
    setSelectedSkillForConfig(skill)
    setShowConfigureModal(true)
  }
  
  // Handle skill deletion
  const handleDeleteSkill = async (skillId: string) => {
    if (!skillId) return
    
    if (!confirm('Are you sure you want to delete this skill? This action cannot be undone.')) {
      return
    }
    
    try {
      await deleteSkillMutation.mutateAsync(skillId)
      toast.success('Skill deleted successfully')
    } catch (error) {
      toast.error('Failed to delete skill')
    }
  }

  // Check if agent has skill
  const agentHasSkill = (skillId: string) => {
    return Array.isArray(agentSkills) ? agentSkills.some(skill => skill.id === skillId) : false
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

        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setShowImportGit(true)}
            className="border-orange-400/50 hover:border-orange-400"
          >
            <Code className="w-4 h-4 mr-2" />
            Import from Git
          </Button>
          <Button
            onClick={() => setShowCreateModal(true)}
            className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Skill
          </Button>
        </div>

      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="all-skills">All Skills</TabsTrigger>
          <TabsTrigger value="skill-categories">Categories</TabsTrigger>
        </TabsList>

        {/* Filters (search removed per request) */}
        <div className="flex flex-col sm:flex-row gap-4">
          <Select value={categoryFilter} onValueChange={setCategoryFilter}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All Categories" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              {availableCategories.map((key) => (
                <SelectItem key={key} value={key}>
                  {key}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Select value={difficultyFilter} onValueChange={setDifficultyFilter}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All Types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="technical">Technical</SelectItem>
              <SelectItem value="cognitive">Cognitive</SelectItem>
              <SelectItem value="analytical">Analytical</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* All Skills Tab (use PRD-22 SkillBrowser for correct cards and detail modal) */}
        <TabsContent value="all-skills" className="space-y-6">
          <SkillBrowser hideUnknownSource />
        </TabsContent>

      {/* Sources tab removed per request; All Skills now lists PRD-22 skills */}

        {/* Agent Skills tab removed per request */}

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

      {/* Skill Configuration Modal */}
      {selectedSkillForConfig && (
        <SkillConfigurationModal
          open={showConfigureModal}
          onClose={() => {
            setShowConfigureModal(false)
            setSelectedSkillForConfig(null)
          }}
          skill={selectedSkillForConfig}
          onSuccess={() => {
            setShowConfigureModal(false)
            setSelectedSkillForConfig(null)
            // Skills will auto-refresh via React Query
          }}
        />
      )}

      {/* PRD-22: Import Git Modal */}
      <ImportGitModal
        open={showImportGit}
        onOpenChange={setShowImportGit}
        onSuccess={() => {
          setShowImportGit(false)
        }}
      />
    </div>
  )
}
