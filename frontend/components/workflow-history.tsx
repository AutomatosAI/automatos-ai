
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  RefreshCw, 
  Eye, 
  Calendar, 
  Github,
  CheckCircle,
  XCircle,
  Clock,
  Loader2
} from 'lucide-react';
import { Workflow } from '@/lib/types';
import { formatDistanceToNow } from 'date-fns';

interface WorkflowHistoryProps {
  workflows: Workflow[];
  onRefresh: () => void;
  onViewWorkflow: (workflow: Workflow) => void;
}

export function WorkflowHistory({ workflows, onRefresh, onViewWorkflow }: WorkflowHistoryProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await onRefresh();
    setIsRefreshing(false);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-400" />;
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-400" />;
      default:
        return <Clock className="h-4 w-4 text-slate-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'running':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'completed':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'failed':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
    }
  };

  const extractRepoName = (url: string) => {
    try {
      const parts = url.split('/');
      return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
    } catch {
      return url;
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h3 className="text-lg font-medium text-white">Recent Workflows</h3>
          <Badge variant="outline" className="border-slate-600 text-slate-400">
            {workflows.length} total
          </Badge>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="border-slate-600 hover:bg-slate-700"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Workflow List */}
      <ScrollArea className="h-96">
        <div className="space-y-3">
          {workflows.length > 0 ? (
            workflows.map((workflow) => (
              <Card
                key={workflow.id}
                className="bg-slate-700/30 border-slate-600 hover:bg-slate-700/50 transition-colors"
              >
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-3 mb-2">
                        {getStatusIcon(workflow.status)}
                        <Badge className={getStatusColor(workflow.status)}>
                          {workflow.status}
                        </Badge>
                        <span className="text-sm text-slate-400">
                          {workflow.progress}%
                        </span>
                      </div>
                      
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2 text-sm">
                          <Github className="h-4 w-4 text-slate-500" />
                          <span className="text-slate-300 truncate">
                            {extractRepoName(workflow.githubUrl)}
                          </span>
                        </div>
                        
                        {workflow.prompt && (
                          <p className="text-xs text-slate-400 truncate">
                            {workflow.prompt}
                          </p>
                        )}
                        
                        <div className="flex items-center space-x-2 text-xs text-slate-500">
                          <Calendar className="h-3 w-3" />
                          <span>
                            {formatDistanceToNow(new Date(workflow.createdAt), { addSuffix: true })}
                          </span>
                          <span>•</span>
                          <span className="capitalize">{workflow.mode}</span>
                        </div>
                      </div>
                    </div>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onViewWorkflow(workflow)}
                      className="text-slate-400 hover:text-white hover:bg-slate-600"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {workflow.error && (
                    <div className="mt-3 p-2 bg-red-500/10 border border-red-500/20 rounded text-xs text-red-400">
                      {workflow.error}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))
          ) : (
            <div className="text-center py-12">
              <Calendar className="h-12 w-12 text-slate-600 mx-auto mb-4" />
              <h4 className="text-lg font-medium text-slate-400 mb-2">No Workflows Yet</h4>
              <p className="text-sm text-slate-500">
                Start your first workflow to see it appear here
              </p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
