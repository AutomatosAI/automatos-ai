'use client'

import { useState, useEffect } from 'react'
import { CheckCircle, AlertCircle, Clock } from 'lucide-react'
import { apiClient } from '@/lib/api'

export function RecentActivities() {
  const [activities, setActivities] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchActivities = async () => {
      try {
        // For now, create activities based on system health
        const health = await apiClient.getSystemHealth()
        setActivities([
          {
            id: 'health-check',
            type: 'system',
            title: 'System Health Check',
            description: 'System health check completed',
            timestamp: health.timestamp || new Date().toISOString(),
            status: 'success'
          }
        ])
      } catch (error) {
        console.error('Failed to fetch activities:', error)
        setActivities([])
      } finally {
        setLoading(false)
      }
    }

    fetchActivities()
    const interval = setInterval(fetchActivities, 60000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Activities</h3>
        <div className="space-y-3">
          <div className="animate-pulse">Loading real activities...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-semibold mb-4">Recent Activities</h3>
      <div className="space-y-3">
        {activities.length === 0 ? (
          <div className="text-muted-foreground text-sm">No recent activities</div>
        ) : (
          activities.map((activity) => (
            <div key={activity.id} className="flex items-start space-x-3 p-3 rounded-lg bg-secondary/20">
              <div className="flex-shrink-0">
                {activity.status === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : activity.status === 'warning' ? (
                  <AlertCircle className="w-5 h-5 text-yellow-400" />
                ) : (
                  <Clock className="w-5 h-5 text-blue-400" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm">{activity.title}</p>
                <p className="text-muted-foreground text-xs mt-1">{activity.description}</p>
                <p className="text-muted-foreground text-xs mt-1">
                  {new Date(activity.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
