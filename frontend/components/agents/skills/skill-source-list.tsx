/**
 * Skill Source List Component
 * ===========================
 * 
 * Displays and manages skill sources (Git repositories, uploads, etc.)
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { 
  GitBranch, 
  RefreshCw, 
  Trash2, 
  CheckCircle2, 
  XCircle, 
  Loader2,
  Package
} from 'lucide-react';
import { useSkillsApi } from '@/hooks/use-skills-api';
import { SkillSource, SkillSourceStatus } from '@/types/skills';

export function SkillSourceList() {
  const { 
    listSkillSources, 
    updateGitRepository, 
    deactivateSkillSource,
    loading 
  } = useSkillsApi();
  
  const [sources, setSources] = useState<SkillSource[]>([]);
  const [deleteConfirm, setDeleteConfirm] = useState<SkillSource | null>(null);

  useEffect(() => {
    loadSources();
  }, []);

  const loadSources = async () => {
    const data = await listSkillSources();
    setSources(data);
  };

  const handleUpdate = async (source: SkillSource) => {
    const success = await updateGitRepository(source.id);
    if (success) {
      loadSources();
    }
  };

  const handleDelete = async (source: SkillSource) => {
    const success = await deactivateSkillSource(source.id, true);
    if (success) {
      setDeleteConfirm(null);
      loadSources();
    }
  };

  const getStatusBadge = (status: SkillSourceStatus) => {
    switch (status) {
      case SkillSourceStatus.ACTIVE:
        return <Badge variant="default" className="bg-green-500">Active</Badge>;
      case SkillSourceStatus.ERROR:
        return <Badge variant="destructive">Error</Badge>;
      case SkillSourceStatus.SYNCING:
        return <Badge variant="secondary">Syncing</Badge>;
      case SkillSourceStatus.INACTIVE:
        return <Badge variant="outline">Inactive</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-4">
      {sources.length === 0 ? (
        <Card>
          <CardContent className="pt-6 text-center">
            <Package className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No skill sources configured</p>
            <p className="text-sm text-muted-foreground mt-2">
              Import skills from a Git repository to get started
            </p>
          </CardContent>
        </Card>
      ) : (
        sources.map(source => (
          <Card key={source.id}>
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="flex items-center gap-2">
                    <GitBranch className="h-5 w-5" />
                    {source.source_name}
                  </CardTitle>
                  <CardDescription className="mt-2">
                    {source.git_url || 'Local source'}
                  </CardDescription>
                </div>
                {getStatusBadge(source.status)}
              </div>
            </CardHeader>

            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Type</p>
                  <p className="font-medium">{source.source_type}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Skills</p>
                  <p className="font-medium">{source.skills_discovered}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Branch</p>
                  <p className="font-medium">{source.git_branch || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Auto-Update</p>
                  <p className="font-medium">{source.auto_update ? 'Enabled' : 'Disabled'}</p>
                </div>
              </div>

              {source.last_sync_at && (
                <div className="mt-4 text-xs text-muted-foreground">
                  Last synced: {new Date(source.last_sync_at).toLocaleString()}
                  {source.last_sync_status === 'success' ? (
                    <CheckCircle2 className="inline ml-2 h-3 w-3 text-green-500" />
                  ) : source.last_sync_status === 'error' ? (
                    <XCircle className="inline ml-2 h-3 w-3 text-red-500" />
                  ) : null}
                </div>
              )}

              {source.last_sync_error && (
                <div className="mt-2 p-2 bg-destructive/10 text-destructive text-xs rounded">
                  {source.last_sync_error}
                </div>
              )}
            </CardContent>

            <CardFooter className="flex gap-2">
              {source.source_type === 'git' && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleUpdate(source)}
                  disabled={loading || source.status === 'syncing'}
                >
                  {loading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-2 h-4 w-4" />
                  )}
                  Update
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setDeleteConfirm(source)}
                disabled={loading}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Remove
              </Button>
            </CardFooter>
          </Card>
        ))
      )}

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteConfirm} onOpenChange={() => setDeleteConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove Skill Source?</AlertDialogTitle>
            <AlertDialogDescription>
              This will deactivate the source "{deleteConfirm?.source_name}" and all associated skills.
              This action can be reversed by re-importing the source.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteConfirm && handleDelete(deleteConfirm)}
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
