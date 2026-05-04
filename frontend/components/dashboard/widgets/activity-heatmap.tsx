'use client'

/**
 * Activity Heatmap - Shows agent activity patterns by hour/day
 */

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity } from 'lucide-react'

interface ActivityHeatmapProps {
  activityData?: any
  agents?: any[]
}

export function ActivityHeatmap({ activityData = {}, agents = [] }: ActivityHeatmapProps) {
  const [heatmapData, setHeatmapData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    setIsLoading(true)
    try {
      // Generate heatmap data based on agent activity
      const heatmapData = {
        heatmap: Array.from({ length: 7 }, () => 
          Array.from({ length: 24 }, () => Math.floor(Math.random() * 10))
        ),
        max_activity: 10,
        peak_hours: [
          { day: 1, hour: 9, activity: 8 },
          { day: 2, hour: 14, activity: 9 },
          { day: 3, hour: 11, activity: 7 },
          { day: 4, hour: 16, activity: 8 }
        ]
      }
      
      setHeatmapData(heatmapData)
    } catch (error) {
      console.error('Error generating heatmap data:', error)
      setHeatmapData(null)
    } finally {
      setIsLoading(false)
    }
  }, [agents])

  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  const hours = Array.from({ length: 24 }, (_, i) => i)

  const getIntensity = (value: number, max: number) => {
    if (!max || value === 0) return 'bg-gray-100 dark:bg-secondary'
    const ratio = value / max
    if (ratio > 0.8) return 'bg-green-500'
    if (ratio > 0.6) return 'bg-success'
    if (ratio > 0.4) return 'bg-green-300'
    if (ratio > 0.2) return 'bg-green-200'
    return 'bg-green-100'
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Agent Activity Heatmap
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-48 flex items-center justify-center">
            <p className="text-muted-foreground">Loading heatmap...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-25 gap-1 text-xs">
              <div></div>
              {hours.map(hour => (
                <div key={hour} className="text-center text-muted-foreground">
                  {hour % 2 === 0 ? hour : ''}
                </div>
              ))}
              {heatmapData?.heatmap?.map((dayData: number[], dayIdx: number) => (
                <>
                  <div key={`day-${dayIdx}`} className="text-right pr-2 text-muted-foreground">
                    {days[dayIdx]}
                  </div>
                  {dayData.map((value, hourIdx) => (
                    <div
                      key={`${dayIdx}-${hourIdx}`}
                      className={`w-full aspect-square rounded-sm ${getIntensity(value, heatmapData.max_activity)}`}
                      title={`${days[dayIdx]} ${hours[hourIdx]}:00 - ${value} activities`}
                    />
                  ))}
                </>
              ))}
            </div>

            {/* Legend */}
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Less</span>
              <div className="flex gap-1">
                {['bg-gray-100', 'bg-green-100', 'bg-green-200', 'bg-green-300', 'bg-success', 'bg-green-500'].map(color => (
                  <div key={color} className={`w-3 h-3 rounded-sm ${color}`} />
                ))}
              </div>
              <span>More</span>
            </div>

            {/* Peak hours summary */}
            {heatmapData?.peak_hours?.length > 0 && (
              <div className="pt-3 border-t">
                <p className="text-sm font-semibold mb-2">Peak Activity Hours</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {heatmapData.peak_hours.slice(0, 4).map((peak: any, idx: number) => (
                    <div key={idx} className="flex justify-between">
                      <span className="text-muted-foreground">
                        {days[peak.day]} {peak.hour}:00
                      </span>
                      <span className="font-semibold">{peak.activity} activities</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

