
'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { useTools, useToolsHealth } from '@/hooks/use-tools-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  Bug, 
  FileText, 
  Activity, 
  Database,
  Play,
  Pause,
  Settings,
  Wrench
} from 'lucide-react'

export default function ToolsPage() {
  const { data: tools, isLoading: toolsLoading } = useTools()
  const { data: toolsHealth, isLoading: healthLoading } = useToolsHealth()

  if (toolsLoading || healthLoading) {
    return (
      <MainLayout>
        <div className="p-6">
          <div className="animate-pulse space-y-6">
            <div className="h-8 bg-secondary rounded w-48"></div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.from({ length: 4 }).map((_, i) => (
                <Card key={i} className="glass-card">
                  <CardHeader>
                    <div className="h-4 bg-secondary rounded w-3/4"></div>
                    <div className="h-3 bg-secondary rounded w-1/2"></div>
                  </CardHeader>
                  <CardContent>
                    <div className="h-16 bg-secondary rounded"></div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </MainLayout>
    )
  }

  const getToolIcon = (iconName: string) => {
    switch (iconName) {
      case 'bug': return Bug
      case 'file-text': return FileText
      case 'activity': return Activity
      case 'database': return Database
      default: return Wrench
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'available': return 'bg-green-500/10 text-green-500 border-green-500/20'
      case 'maintenance': return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
      case 'error': return 'bg-red-500/10 text-red-500 border-red-500/20'
      default: return 'bg-gray-500/10 text-gray-500 border-gray-500/20'
    }
  }

  return (
    <MainLayout>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Tools Dashboard</h1>
            <p className="text-muted-foreground">
              Development and utility tools for system management
            </p>
          </div>
          {toolsHealth && (
            <div className="text-right">
              <div className="text-sm text-muted-foreground">System Health</div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-green-500 border-green-500/20">
                  {toolsHealth.status}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  {toolsHealth.available}/{toolsHealth.total} available
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tools && tools.length > 0 ? tools.map((tool: any) => {
            const Icon = getToolIcon(tool.icon || 'wrench')
            return (
              <Card key={tool.id} className="glass-card hover:shadow-lg transition-shadow">
                <CardHeader className="flex flex-row items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Icon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{tool.name}</CardTitle>
                      <p className="text-sm text-muted-foreground capitalize">
                        {tool.type?.replace('_', ' ')}
                      </p>
                    </div>
                  </div>
                  <Badge 
                    variant="outline" 
                    className={getStatusColor(tool.status)}
                  >
                    {tool.status}
                  </Badge>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    {tool.description}
                  </p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" className="flex-1">
                      <Play className="w-4 h-4 mr-2" />
                      Start
                    </Button>
                    <Button size="sm" variant="ghost">
                      <Settings className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )
          }) : (
            <div className="col-span-full text-center py-12">
              <Wrench className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Tools Available</h3>
              <p className="text-muted-foreground">
                No tools are currently available in the system.
              </p>
            </div>
          )}
        </div>
      </div>
    </MainLayout>
  )
}
