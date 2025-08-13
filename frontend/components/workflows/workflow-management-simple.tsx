'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { 
  Play, 
  Plus, 
  Search, 
  Filter,
  GitBranch,
  Clock,
  CheckCircle,
  AlertTriangle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient } from '@/lib/api'
import { CreateWorkflowModal } from './create-workflow-modal'
import { RunWorkflowModal } from './run-workflow-modal'
import { EditWorkflowModal } from './edit-workflow-modal'

export function WorkflowManagement() {
  const [searchTerm, setSearchTerm] = useState('')
  const [workflows, setWorkflows] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [editId, setEditId] = useState<string | null>(null)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const stats = [
    {
      label: 'Active Workflows',
      value: '1',
      change: '+1 total',
      icon: GitBranch,
      color: 'text-blue-400'
    },
    {
      label: 'Completed Today',
      value: '2',
      change: '33% success rate',
      icon: CheckCircle,
      color: 'text-green-400'
    },
    {
      label: 'Avg Duration',
      value: '2.4h',
      change: 'Based on history',
      icon: Clock,
      color: 'text-orange-400'
    },
    {
      label: 'Agent Utilization',
      value: '0%',
      change: 'CPU usage based',
      icon: Play,
      color: 'text-purple-400'
    }
  ]

  // Load workflows from backend
  useEffect(() => {
    const loadWorkflows = async () => {
      try {
        setLoading(true)
        setError(null)
        const workflowsData = await apiClient.getWorkflows()
        setWorkflows(workflowsData)
      } catch (err) {
        console.error('Error loading workflows:', err)
        setError(err instanceof Error ? err.message : 'Failed to load workflows')
      } finally {
        setLoading(false)
      }
    }

    loadWorkflows()
    const h = () => loadWorkflows()
    window.addEventListener('workflows:refresh', h)
    return () => window.removeEventListener('workflows:refresh', h)
  }, [])

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Workflow <span className="gradient-text">Management</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Create, monitor, and manage your multi-agent workflows
          </p>
        </div>
        
        <Button className="gradient-accent hover:opacity-90 transition-opacity" onClick={()=>setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Create Workflow
        </Button>
      </motion.div>

      {/* Stats Overview */}
      <motion.div
        ref={ref}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stat.value}</h3>
              <p className="text-muted-foreground text-sm">{stat.label}</p>
              <p className="text-xs text-green-400">{stat.change}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Workflow Management */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="active" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="active" className="flex items-center space-x-2">
              <Play className="w-4 h-4" />
              <span className="hidden sm:inline">Active</span>
            </TabsTrigger>
            <TabsTrigger value="templates" className="flex items-center space-x-2">
              <GitBranch className="w-4 h-4" />
              <span className="hidden sm:inline">Templates</span>
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center space-x-2">
              <Clock className="w-4 h-4" />
              <span className="hidden sm:inline">History</span>
            </TabsTrigger>
            <TabsTrigger value="monitoring" className="flex items-center space-x-2">
              <Play className="w-4 h-4" />
              <span className="hidden sm:inline">Monitoring</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-6">
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search workflows by name, category, or tags..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
                />
              </div>
              <Button variant="outline" className="shrink-0">
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
            </div>

            {/* Loading State */}
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading workflows...</p>
                </div>
              </div>
            )}

            {/* Error State */}
            {error && !loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error loading workflows: {error}</p>
                  <Button variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!loading && !error && workflows.length === 0 && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <GitBranch className="h-8 w-8 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground mb-4">No workflows created yet</p>
                  <Button className="gradient-accent hover:opacity-90">
                    <Plus className="w-4 h-4 mr-2" />
                    Create Your First Workflow
                  </Button>
                </div>
              </div>
            )}

            {/* Workflow List */}
            {!loading && !error && workflows.length > 0 && (
              <div className="space-y-4">
                {workflows.filter(w => !searchTerm || (w.name||'').toLowerCase().includes(searchTerm.toLowerCase())).map((workflow, index) => (
                  <Card key={workflow.id} className="glass-card">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold text-lg">{workflow.name || 'Unnamed Workflow'}</h3>
                          <p className="text-sm text-muted-foreground">
                            {workflow.description || 'No description'}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20">{workflow.status || 'unknown'}</Badge>
                          <Button size="sm" variant="secondary" onClick={()=>setRunId(String(workflow.id))}>Run</Button>
                          <Button size="sm" variant="outline" onClick={()=>setEditId(String(workflow.id))}>Edit</Button>
                          <Button size="sm" variant="destructive" onClick={async ()=>{
                            if (!confirm('Delete this workflow?')) return
                            await apiClient.deleteWorkflow(workflow.id)
                            try { window.dispatchEvent(new Event('workflows:refresh')) } catch {}
                          }}>Delete</Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="templates" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Workflow Templates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Workflow templates will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Workflow History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Workflow execution history will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Real-time Monitoring</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Real-time workflow monitoring dashboard will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        <CreateWorkflowModal open={createOpen} onClose={()=>setCreateOpen(false)} />
        {runId ? <RunWorkflowModal open={!!runId} onClose={()=>setRunId(null)} id={runId} /> : null}
        {editId ? <EditWorkflowModal open={!!editId} onClose={()=>setEditId(null)} id={editId} /> : null}
      </motion.div>
    </div>
  )
}
