'use client'

import { motion } from 'framer-motion'
import { X, Brain, Database, Users, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'

interface MemoryVisualizationProps {
  workflowId: number
  agents: any[]
  onClose: () => void
}

export function MemoryVisualization({
  workflowId,
  agents,
  onClose
}: MemoryVisualizationProps) {
  return (
    <motion.div
      initial={{ x: '100%' }}
      animate={{ x: 0 }}
      exit={{ x: '100%' }}
      transition={{ type: 'spring', damping: 20 }}
      className="fixed right-0 top-0 h-full w-[500px] bg-background border-l border-border shadow-2xl z-50"
    >
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border/50">
          <div className="flex items-center space-x-2">
            <Brain className="w-5 h-5 text-agent" />
            <h2 className="font-semibold">Memory Network</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Content */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {/* Collective Memory (Shared) */}
            <Card className="glass-card border-agent/20">
              <CardHeader>
                <CardTitle className="text-sm flex items-center justify-between">
                  <span className="flex items-center">
                    <Users className="w-4 h-4 mr-2 text-agent" />
                    Collective Memory
                  </span>
                  <Badge className="bg-agent/10 text-agent">
                    Shared
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-xs text-muted-foreground mb-2">
                  Accessible by all agents in the workflow
                </div>

                <div className="space-y-2">
                  {/* Example shared memories */}
                  <MemoryItem
                    key="api_patterns"
                    name="api_patterns"
                    type="pattern"
                    source="CodeArchitect"
                    accessCount={3}
                  />
                  <MemoryItem
                    key="security_rules"
                    name="security_rules"
                    type="knowledge"
                    source="SecurityExpert"
                    accessCount={2}
                  />
                  <MemoryItem
                    key="test_strategies"
                    name="test_strategies"
                    type="strategy"
                    source="TestMaster"
                    accessCount={1}
                  />
                </div>

                <div className="flex items-center justify-between text-xs pt-2 border-t border-border/50">
                  <span className="text-muted-foreground">Total Items</span>
                  <span className="font-medium">24 patterns</span>
                </div>
              </CardContent>
            </Card>

            {/* Individual Agent Memories */}
            {agents.map((agent: any) => (
              <Card key={agent.id} className="glass-card">
                <CardHeader>
                  <CardTitle className="text-sm flex items-center justify-between">
                    <span className="flex items-center">
                      <Brain className="w-4 h-4 mr-2 text-info" />
                      {agent.name}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {agent.agent_type}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Memory Hierarchy */}
                  <div className="space-y-2">
                    <MemoryLevel
                      name="Working Memory"
                      count={7}
                      capacity={7}
                      color="blue"
                      icon={<TrendingUp className="w-3 h-3" />}
                    />
                    <MemoryLevel
                      name="Short-term Memory"
                      count={42}
                      capacity={100}
                      color="green"
                      icon={<Database className="w-3 h-3" />}
                    />
                    <MemoryLevel
                      name="Long-term Memory"
                      count={156}
                      capacity="∞"
                      color="purple"
                      icon={<Brain className="w-3 h-3" />}
                    />
                  </div>

                  {/* Recent Operations */}
                  <div className="pt-2 border-t border-border/50">
                    <div className="text-xs text-muted-foreground mb-2">
                      Recent Operations
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-success">✓ Stored</span>
                        <span className="text-muted-foreground">current_analysis</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-info">→ Retrieved</span>
                        <span className="text-muted-foreground">api_patterns</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}

            {/* Memory Stats */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-sm">Memory Statistics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Memories:</span>
                  <span className="font-medium">287</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Shared Items:</span>
                  <span className="font-medium">24</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Operations/min:</span>
                  <span className="font-medium">12</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Storage Used:</span>
                  <span className="font-medium text-success">2.3 GB</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      </div>
    </motion.div>
  )
}

function MemoryItem({
  name,
  type,
  source,
  accessCount
}: {
  name: string
  type: string
  source: string
  accessCount: number
}) {
  return (
    <div className="flex items-center justify-between p-2 rounded bg-background/50 text-xs">
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate">{name}</div>
        <div className="text-muted-foreground">
          <span className="text-agent">{source}</span> • {type}
        </div>
      </div>
      <Badge variant="outline" className="text-xs ml-2">
        {accessCount}x
      </Badge>
    </div>
  )
}

function MemoryLevel({
  name,
  count,
  capacity,
  color,
  icon
}: {
  name: string
  count: number
  capacity: number | string
  color: string
  icon: React.ReactNode
}) {
  const percentage = typeof capacity === 'number' ? (count / capacity) * 100 : 0

  const colorClasses = {
    blue: 'text-info bg-info/10',
    green: 'text-success bg-success/10',
    purple: 'text-agent bg-agent/10'
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center space-x-1">
          {icon}
          <span>{name}</span>
        </span>
        <span className="font-medium">{count} {capacity !== '∞' && `/ ${capacity}`}</span>
      </div>
      {capacity !== '∞' && (
        <div className="h-1 bg-secondary rounded-full overflow-hidden">
          <div 
            className={`h-full ${colorClasses[color as keyof typeof colorClasses]} transition-all duration-300`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      )}
    </div>
  )
}

