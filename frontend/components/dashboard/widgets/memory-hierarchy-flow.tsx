'use client'

/**
 * Memory Hierarchy Flow - Sankey diagram of memory consolidation
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Brain, ArrowRight } from 'lucide-react'

interface MemoryData {
  total_memories: number
  working_memory_count: number
  short_term_count: number
  long_term_count: number
  consolidation_rate: number
  knowledge_nodes: number
  knowledge_edges: number
  avg_importance_score: number
  memory_growth_rate: number
}

interface MemoryHierarchyFlowProps {
  data: MemoryData
}

export function MemoryHierarchyFlow({ data }: MemoryHierarchyFlowProps) {
  const workingToShortRate = data.working_memory_count > 0
    ? ((data.short_term_count / data.working_memory_count) * 100).toFixed(1)
    : '0'
  
  const shortToLongRate = data.short_term_count > 0
    ? ((data.long_term_count / data.short_term_count) * 100).toFixed(1)
    : '0'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Memory Hierarchy Flow
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Flow Visualization */}
          <div className="relative">
            <div className="flex items-center justify-between">
              {/* Working Memory */}
              <div className="text-center flex-1">
                <div className="w-20 h-20 mx-auto rounded-full bg-info/10 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-info">
                      {data.working_memory_count}
                    </p>
                  </div>
                </div>
                <p className="text-sm mt-2">Working</p>
                <p className="text-xs text-muted-foreground">Immediate</p>
              </div>

              {/* Arrow 1 */}
              <div className="flex-1 flex flex-col items-center justify-center px-2">
                <ArrowRight className="w-6 h-6 text-muted-foreground" />
                <p className="text-xs text-muted-foreground mt-1">{workingToShortRate}%</p>
              </div>

              {/* Short-term Memory */}
              <div className="text-center flex-1">
                <div className="w-20 h-20 mx-auto rounded-full bg-success/10 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-success">
                      {data.short_term_count}
                    </p>
                  </div>
                </div>
                <p className="text-sm mt-2">Short-term</p>
                <p className="text-xs text-muted-foreground">Temporary</p>
              </div>

              {/* Arrow 2 */}
              <div className="flex-1 flex flex-col items-center justify-center px-2">
                <ArrowRight className="w-6 h-6 text-muted-foreground" />
                <p className="text-xs text-muted-foreground mt-1">{shortToLongRate}%</p>
              </div>

              {/* Long-term Memory */}
              <div className="text-center flex-1">
                <div className="w-20 h-20 mx-auto rounded-full bg-agent/10 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-agent">
                      {data.long_term_count}
                    </p>
                  </div>
                </div>
                <p className="text-sm mt-2">Long-term</p>
                <p className="text-xs text-muted-foreground">Permanent</p>
              </div>
            </div>
          </div>

          {/* Consolidation Metrics */}
          <div className="grid grid-cols-2 gap-4 pt-4 border-t">
            <div>
              <p className="text-xs text-muted-foreground">Consolidation Rate</p>
              <p className="text-lg font-semibold">{data.consolidation_rate.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Avg Importance</p>
              <p className="text-lg font-semibold">{data.avg_importance_score.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Growth Rate</p>
              <p className="text-lg font-semibold">
                {data.memory_growth_rate > 0 ? '+' : ''}{data.memory_growth_rate.toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Knowledge Density</p>
              <p className="text-lg font-semibold">
                {data.knowledge_nodes > 0 
                  ? (data.knowledge_edges / data.knowledge_nodes).toFixed(2)
                  : '0'}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

