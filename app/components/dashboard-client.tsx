'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Terminal, Activity } from 'lucide-react';
import { WorkflowForm } from '@/components/workflow-form';
import { ProgressPanel } from '@/components/progress-panel';
import { WorkflowHistory } from '@/components/workflow-history';
import { Workflow } from '@/lib/types';
import { toast } from 'sonner';

export default function DashboardClient() {
  const [activeWorkflow, setActiveWorkflow] = useState<Workflow | null>(null);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchWorkflows = async () => {
    // Temporarily disabled - just return empty array
    setWorkflows([]);
  };

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const handleWorkflowStart = (workflow: Workflow) => {
    setActiveWorkflow(workflow);
    setWorkflows(prev => [workflow, ...prev]);
    toast.success('Workflow started successfully!');
  };

  const handleWorkflowSelect = (workflow: Workflow) => {
    setActiveWorkflow(workflow);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-600 rounded-lg">
              <Terminal className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Automotas AI</h1>
              <p className="text-slate-400">Multi-Agent Orchestration Platform</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Button
              variant="outline"
              size="sm"
              onClick={fetchWorkflows}
              className="border-slate-600 text-slate-300 hover:bg-slate-800"
            >
              <Activity className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Workflow Form */}
          <Card className="lg:col-span-1 bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Start New Workflow</CardTitle>
              <CardDescription className="text-slate-400">
                Configure and launch a new automation workflow
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

          {/* Middle Column - Progress Panel */}
          <Card className="lg:col-span-1 bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Workflow Progress</CardTitle>
              <CardDescription className="text-slate-400">
                Real-time status and logs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ProgressPanel 
                workflow={activeWorkflow}
              />
            </CardContent>
          </Card>

          {/* Right Column - Workflow History */}
          <Card className="lg:col-span-1 bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Workflow History</CardTitle>
              <CardDescription className="text-slate-400">
                Recent workflow executions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <WorkflowHistory
                workflows={workflows}
                onWorkflowSelect={handleWorkflowSelect}
                activeWorkflow={activeWorkflow}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
