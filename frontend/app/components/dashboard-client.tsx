
'use client';

import { useState, useEffect } from 'react';
import { useSession, signOut } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LogOut, Terminal, Activity } from 'lucide-react';
import { WorkflowForm } from '@/components/workflow-form';
import { ProgressPanel } from '@/components/progress-panel';
import { WorkflowHistory } from '@/components/workflow-history';
import { Workflow } from '@/lib/types';
import { toast } from 'sonner';

export default function DashboardClient() {
  const { data: session } = useSession();
  const [activeWorkflow, setActiveWorkflow] = useState<Workflow | null>(null);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('/api/workflow');
      const result = await response.json();
      
      if (result.success) {
        setWorkflows(result.data || []);
      }
    } catch (error) {
      console.error('Error fetching workflows:', error);
    }
  };

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const handleWorkflowStart = async (workflow: Workflow) => {
    setActiveWorkflow(workflow);
    await fetchWorkflows();
    toast.success('Workflow started successfully');
  };

  const handleSignOut = () => {
    signOut({ callbackUrl: '/login' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-700 bg-slate-800/80 backdrop-blur-sm">
        <div className="container mx-auto max-w-6xl flex h-16 items-center justify-between px-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-600 rounded-lg">
              <Terminal className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">DevOps Orchestrator</h1>
              <p className="text-xs text-slate-400">Multi-Agent System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm font-medium text-white">{session?.user?.name || session?.user?.email}</p>
              <p className="text-xs text-slate-400">Administrator</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleSignOut}
              className="border-slate-600 hover:bg-slate-700"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto max-w-6xl p-4 space-y-6">
        {/* Welcome Section */}
        <div className="text-center py-8">
          <h2 className="text-3xl font-bold text-white mb-2">
            Welcome to the Orchestration Dashboard
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Manage and monitor your multi-agent workflows. Submit GitHub repositories for automated analysis, 
            review, and deployment through our intelligent agent system.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Workflow Form */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Activity className="h-5 w-5 mr-2 text-purple-400" />
                New Workflow
              </CardTitle>
              <CardDescription className="text-slate-400">
                Start a new multi-agent workflow by providing a GitHub repository
              </CardDescription>
            </CardHeader>
            <CardContent>
              <WorkflowForm
                onWorkflowStart={handleWorkflowStart}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
            </CardContent>
          </Card>

          {/* Progress Panel */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Real-time Progress</CardTitle>
              <CardDescription className="text-slate-400">
                Monitor active workflow progress and agent activities
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ProgressPanel activeWorkflow={activeWorkflow} />
            </CardContent>
          </Card>
        </div>

        {/* Workflow History */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Workflow History</CardTitle>
            <CardDescription className="text-slate-400">
              Review past workflows and their results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <WorkflowHistory 
              workflows={workflows} 
              onRefresh={fetchWorkflows}
              onViewWorkflow={setActiveWorkflow}
            />
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
