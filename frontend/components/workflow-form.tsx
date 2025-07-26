
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Github, Code, MessageSquare } from 'lucide-react';
import { WorkflowFormData, Workflow } from '@/lib/types';
import { toast } from 'sonner';

interface WorkflowFormProps {
  onWorkflowStart: (workflow: Workflow) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export function WorkflowForm({ onWorkflowStart, isLoading, setIsLoading }: WorkflowFormProps) {
  const [formData, setFormData] = useState<WorkflowFormData>({
    githubUrl: '',
    prompt: '',
    mode: 'self-contained',
  });
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      // Validate GitHub URL
      if (!formData.githubUrl.includes('github.com')) {
        throw new Error('Please provide a valid GitHub URL');
      }

      // Validate prompt for prompt-driven mode
      if (formData.mode === 'prompt-driven' && !formData.prompt?.trim()) {
        throw new Error('Prompt is required for prompt-driven mode');
      }

      const response = await fetch('/api/workflow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repository_url: formData.githubUrl,
          task_prompt: formData.prompt || undefined,
        }),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to start workflow');
      }

      // Create a mock workflow object for the callback
      const mockWorkflow: Workflow = {
        id: result.data.workflowId,
        githubUrl: formData.githubUrl,
        prompt: formData.prompt,
        mode: formData.mode,
        status: 'pending',
        progress: 0,
        userId: '',
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      onWorkflowStart(mockWorkflow);

      // Reset form
      setFormData({
        githubUrl: '',
        prompt: '',
        mode: 'self-contained',
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An error occurred';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModeChange = (checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      mode: checked ? 'prompt-driven' : 'self-contained',
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {error && (
        <Alert className="border-red-500 bg-red-500/10">
          <AlertDescription className="text-red-400">
            {error}
          </AlertDescription>
        </Alert>
      )}

      {/* GitHub URL Input */}
      <div className="space-y-2">
        <Label htmlFor="githubUrl" className="text-slate-300 flex items-center">
          <Github className="h-4 w-4 mr-2" />
          GitHub Repository URL
        </Label>
        <Input
          id="githubUrl"
          type="url"
          placeholder="https://github.com/username/repository"
          value={formData.githubUrl}
          onChange={(e) => setFormData(prev => ({ ...prev, githubUrl: e.target.value }))}
          required
          className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
        />
      </div>

      {/* Mode Toggle */}
      <Card className="bg-slate-700/50 border-slate-600">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Code className="h-5 w-5 text-purple-400" />
              <div>
                <Label className="text-slate-300 font-medium">
                  {formData.mode === 'self-contained' ? 'Self-contained Analysis' : 'Prompt-driven Mode'}
                </Label>
                <p className="text-sm text-slate-400">
                  {formData.mode === 'self-contained' 
                    ? 'Automatic analysis and recommendations' 
                    : 'Custom task with specific instructions'
                  }
                </p>
              </div>
            </div>
            <Switch
              checked={formData.mode === 'prompt-driven'}
              onCheckedChange={handleModeChange}
              className="data-[state=checked]:bg-purple-600"
            />
          </div>
        </CardContent>
      </Card>

      {/* Prompt Input (conditional) */}
      {formData.mode === 'prompt-driven' && (
        <div className="space-y-2">
          <Label htmlFor="prompt" className="text-slate-300 flex items-center">
            <MessageSquare className="h-4 w-4 mr-2" />
            Task Instructions
          </Label>
          <Textarea
            id="prompt"
            placeholder="Describe what you want the agents to do with this repository..."
            value={formData.prompt}
            onChange={(e) => setFormData(prev => ({ ...prev, prompt: e.target.value }))}
            rows={4}
            className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-400 resize-none"
          />
          <p className="text-xs text-slate-500">
            Provide specific instructions for the multi-agent system to follow
          </p>
        </div>
      )}

      {/* Submit Button */}
      <Button
        type="submit"
        className="w-full bg-purple-600 hover:bg-purple-700 text-white"
        disabled={isLoading}
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Starting Workflow...
          </>
        ) : (
          'Start Workflow'
        )}
      </Button>
    </form>
  );
}
