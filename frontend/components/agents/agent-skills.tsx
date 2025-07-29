
'use client'

import { useState } from 'react'
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
  BarChart
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

const skillCategories = {
  development: {
    name: 'Development',
    icon: Code,
    color: 'text-blue-400',
    skills: [
      { 
        name: 'Code Analysis', 
        description: 'Analyze code structure, patterns, and quality metrics',
        agents: 8,
        difficulty: 'intermediate',
        tags: ['analysis', 'quality', 'metrics']
      },
      { 
        name: 'Debugging', 
        description: 'Identify and resolve bugs in codebases',
        agents: 6,
        difficulty: 'advanced',
        tags: ['debugging', 'troubleshooting', 'errors']
      },
      { 
        name: 'Refactoring', 
        description: 'Improve code structure while maintaining functionality',
        agents: 5,
        difficulty: 'advanced',
        tags: ['optimization', 'structure', 'maintenance']
      },
      { 
        name: 'Test Generation', 
        description: 'Create comprehensive test suites for applications',
        agents: 4,
        difficulty: 'intermediate',
        tags: ['testing', 'quality', 'automation']
      }
    ]
  },
  security: {
    name: 'Security',
    icon: Shield,
    color: 'text-red-400',
    skills: [
      { 
        name: 'Vulnerability Scanning', 
        description: 'Detect security vulnerabilities in code and systems',
        agents: 3,
        difficulty: 'advanced',
        tags: ['security', 'scanning', 'vulnerabilities']
      },
      { 
        name: 'Threat Modeling', 
        description: 'Analyze potential security threats and risks',
        agents: 2,
        difficulty: 'expert',
        tags: ['threats', 'modeling', 'risk']
      },
      { 
        name: 'Compliance Check', 
        description: 'Ensure adherence to security standards and regulations',
        agents: 3,
        difficulty: 'intermediate',
        tags: ['compliance', 'standards', 'regulations']
      }
    ]
  },
  infrastructure: {
    name: 'Infrastructure',
    icon: Database,
    color: 'text-green-400',
    skills: [
      { 
        name: 'Deployment', 
        description: 'Automated deployment and release management',
        agents: 5,
        difficulty: 'intermediate',
        tags: ['deployment', 'automation', 'ci-cd']
      },
      { 
        name: 'Scaling', 
        description: 'Dynamic resource scaling based on demand',
        agents: 3,
        difficulty: 'advanced',
        tags: ['scaling', 'resources', 'optimization']
      },
      { 
        name: 'Monitoring', 
        description: 'Real-time system and application monitoring',
        agents: 6,
        difficulty: 'intermediate',
        tags: ['monitoring', 'alerts', 'metrics']
      }
    ]
  },
  analytics: {
    name: 'Analytics',
    icon: BarChart,
    color: 'text-purple-400',
    skills: [
      { 
        name: 'Data Processing', 
        description: 'Process and transform large datasets',
        agents: 4,
        difficulty: 'intermediate',
        tags: ['data', 'processing', 'transformation']
      },
      { 
        name: 'Pattern Recognition', 
        description: 'Identify patterns and trends in data',
        agents: 3,
        difficulty: 'advanced',
        tags: ['patterns', 'recognition', 'analysis']
      },
      { 
        name: 'Report Generation', 
        description: 'Generate comprehensive analytical reports',
        agents: 5,
        difficulty: 'intermediate',
        tags: ['reports', 'analytics', 'insights']
      }
    ]
  }
}

const difficultyColors: Record<string, string> = {
  beginner: 'bg-green-500/10 text-green-400 border-green-500/20',
  intermediate: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  advanced: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  expert: 'bg-red-500/10 text-red-400 border-red-500/20'
}

export function AgentSkills() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Skills Management</h2>
          <p className="text-muted-foreground">
            Manage skills and capabilities across your agent ecosystem
          </p>
        </div>
        <Button className="gradient-accent hover:opacity-90">
          <Plus className="w-4 h-4 mr-2" />
          Create Skill
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search skills..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
          />
        </div>
      </div>

      {/* Skills by Category */}
      <Tabs defaultValue="all" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid bg-secondary/50">
          <TabsTrigger value="all">All Skills</TabsTrigger>
          <TabsTrigger value="development">Development</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="infrastructure">Infrastructure</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-6">
          {Object.entries(skillCategories).map(([key, category]) => (
            <SkillCategorySection key={key} category={category} searchTerm={searchTerm} />
          ))}
        </TabsContent>

        {Object.entries(skillCategories).map(([key, category]) => (
          <TabsContent key={key} value={key} className="space-y-6">
            <SkillCategorySection category={category} searchTerm={searchTerm} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}

function SkillCategorySection({ category, searchTerm }: { category: any, searchTerm: string }) {
  const filteredSkills = category.skills.filter((skill: any) =>
    skill.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    skill.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    skill.tags.some((tag: string) => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  if (filteredSkills.length === 0 && searchTerm) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-secondary/50 flex items-center justify-center">
              <category.icon className={`w-4 h-4 ${category.color}`} />
            </div>
            <span>{category.name}</span>
            <Badge variant="secondary">{filteredSkills.length} skills</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredSkills.map((skill: any, index: number) => (
              <motion.div
                key={skill.name}
                className="p-4 rounded-lg border border-border/50 hover:border-primary/20 transition-all duration-300 hover:bg-secondary/20"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className="flex items-start justify-between mb-3">
                  <h4 className="font-semibold text-sm">{skill.name}</h4>
                  <div className="flex space-x-1">
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-red-400 hover:text-red-300">
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
                
                <p className="text-xs text-muted-foreground mb-3">
                  {skill.description}
                </p>
                
                <div className="flex items-center justify-between mb-3">
                  <Badge className={difficultyColors[skill.difficulty]}>
                    {skill.difficulty}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {skill.agents} agents
                  </span>
                </div>
                
                <div className="flex flex-wrap gap-1">
                  {skill.tags.map((tag: string) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
