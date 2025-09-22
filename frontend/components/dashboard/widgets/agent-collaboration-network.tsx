'use client'

/**
 * Agent Collaboration Network - Interactive network graph
 */

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Users, Network } from 'lucide-react'

interface CollaborationData {
  total_sessions: number
  active_sessions: number
  avg_agents_per_session: number
  shared_contexts_count: number
  avg_access_count: number
  collaboration_success_rate: number
  message_exchange_count: number
}

interface AgentCollaborationNetworkProps {
  data: CollaborationData
}

export function AgentCollaborationNetwork({ data }: AgentCollaborationNetworkProps) {
  const [networkData, setNetworkData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const fetchNetworkData = async () => {
      setIsLoading(true)
      try {
        const response = await fetch('/api/analytics/collaboration')
        const result = await response.json()
        setNetworkData(result.network)
      } catch (error) {
        console.error('Error fetching network data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchNetworkData()
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="w-5 h-5" />
          Agent Collaboration Network
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-64 flex items-center justify-center">
            <p className="text-muted-foreground">Loading network data...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="h-64 border rounded-lg flex items-center justify-center bg-muted/10">
              <p className="text-sm text-muted-foreground">
                Network visualization with {networkData?.nodes?.length || 0} nodes
                and {networkData?.edges?.length || 0} connections
              </p>
            </div>

            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold">{data.active_sessions}</p>
                <p className="text-xs text-muted-foreground">Active Sessions</p>
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {data.avg_agents_per_session.toFixed(1)}
                </p>
                <p className="text-xs text-muted-foreground">Avg Agents/Session</p>
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {data.collaboration_success_rate.toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground">Success Rate</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

