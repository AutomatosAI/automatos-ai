
'use client'

import { motion } from 'framer-motion'
import { formatDistanceToNow } from 'date-fns'
import { 
  Bot, 
  FileText, 
  GitBranch, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  Zap,
  Upload,
  Play,
  Square
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useSystemActivities } from '@/hooks/use-system-config-api'

interface ActivityItem {
  id: string
  type: string
  title: string
  description?: string
  timestamp: string
  status: 'success' | 'error' | 'warning' | 'info'
  metadata?: Record<string, any>
}

export function ActivityFeed() {
  // Use real API hook
  const { data: activities, isLoading } = useSystemActivities()
  const getActivityIcon = (type: string, status: string) => {
    if (status === 'error') return AlertCircle
    if (status === 'success') return CheckCircle
    
    switch (type) {
      case 'agent':
        return Bot
      case 'document':
        return FileText
      case 'workflow':
        return GitBranch
      case 'upload':
        return Upload
      case 'execution':
        return Play
      case 'system':
        return Zap
      default:
        return Clock
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'text-success'
      case 'error':
        return 'text-destructive'
      case 'warning':
        return 'text-warning'
      default:
        return 'text-info'
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success':
        return 'default'
      case 'error':
        return 'destructive'
      case 'warning':
        return 'secondary'
      default:
        return 'outline'
    }
  }

  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-brand-accent" />
            Recent Activities
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-start gap-3 p-3">
                <Skeleton className="w-8 h-8 rounded-full" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
                <Skeleton className="h-5 w-12" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  // Process real activities data - ensure it's always an array
  const processedActivities: ActivityItem[] = Array.isArray(activities) ? activities : []
  const displayActivities = processedActivities.slice(0, 8)

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="w-5 h-5 text-brand-accent" />
          Recent Activities
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-80 px-6">
          <div className="space-y-1">
            {displayActivities.map((activity, index) => {
              const Icon = getActivityIcon(activity.type, activity.status)
              
              return (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="flex items-start gap-3 p-3 rounded-lg hover:bg-accent/50 transition-colors"
                >
                  <div className={`p-2 rounded-full bg-background ${getStatusColor(activity.status)}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="text-sm font-medium truncate">{activity.title}</p>
                      <Badge variant={getStatusBadge(activity.status) as any} className="text-xs">
                        {activity.status}
                      </Badge>
                    </div>
                    
                    {activity.description && (
                      <p className="text-xs text-muted-foreground truncate">
                        {activity.description}
                      </p>
                    )}
                    
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                    </p>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
