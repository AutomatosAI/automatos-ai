/**
 * Git Import Modal Component
 * ==========================
 * 
 * Modal for importing skills from Git repositories.
 * 
 * Features:
 * - Git URL input with validation
 * - Branch selection
 * - Auto-update configuration
 * - Real-time import progress
 */

'use client';

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Loader2, GitBranch, CheckCircle2, XCircle } from 'lucide-react';
import { useSkillsApi } from '@/hooks/use-skills-api';
import { GitImportRequest } from '@/types/skills';

interface ImportGitModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

export function ImportGitModal({ open, onOpenChange, onSuccess }: ImportGitModalProps) {
  const { importGitRepository, loading } = useSkillsApi();
  
  const [gitUrl, setGitUrl] = useState('');
  const [sourceName, setSourceName] = useState('');
  const [branch, setBranch] = useState('main');
  const [autoUpdate, setAutoUpdate] = useState(false);
  const [importResult, setImportResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const data: GitImportRequest = {
      git_url: gitUrl,
      source_name: sourceName || undefined,
      branch,
      auto_update: autoUpdate,
      update_frequency_hours: autoUpdate ? 24 : undefined
    };

    const result = await importGitRepository(data);
    
    if (result) {
      setImportResult(result);
      if (onSuccess) {
        onSuccess();
      }
      
      // Reset form after success
      setTimeout(() => {
        setGitUrl('');
        setSourceName('');
        setBranch('main');
        setAutoUpdate(false);
        setImportResult(null);
        onOpenChange(false);
      }, 3000);
    }
  };

  const handleClose = () => {
    setGitUrl('');
    setSourceName('');
    setBranch('main');
    setAutoUpdate(false);
    setImportResult(null);
    onOpenChange(false);
  };

  // Auto-generate source name from Git URL
  const autoGenerateSourceName = () => {
    if (gitUrl && !sourceName) {
      const match = gitUrl.match(/\/([^\/]+?)(?:\.git)?$/);
      if (match) {
        setSourceName(`git-${match[1]}`);
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitBranch className="h-5 w-5" />
            Import Skills from Git Repository
          </DialogTitle>
          <DialogDescription>
            Clone a Git repository containing skills and automatically index all SKILL.md files.
          </DialogDescription>
        </DialogHeader>

        {importResult ? (
          <div className="py-8 text-center space-y-4">
            <div className="flex justify-center">
              <CheckCircle2 className="h-16 w-16 text-green-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">{importResult.message}</h3>
              <p className="text-sm text-muted-foreground mt-2">
                Source: <strong>{importResult.source_name}</strong>
              </p>
              <p className="text-sm text-muted-foreground">
                Skills discovered: <strong>{importResult.skills_discovered}</strong>
              </p>
              {importResult.commit_sha && (
                <p className="text-xs text-muted-foreground mt-1">
                  Commit: {importResult.commit_sha.substring(0, 8)}
                </p>
              )}
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Git URL */}
            <div className="space-y-2">
              <Label htmlFor="git_url">
                Git Repository URL <span className="text-red-500">*</span>
              </Label>
              <Input
                id="git_url"
                type="url"
                placeholder="https://github.com/anthropics/skills.git"
                value={gitUrl}
                onChange={(e) => setGitUrl(e.target.value)}
                onBlur={autoGenerateSourceName}
                required
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                Example: https://github.com/anthropics/skills.git
              </p>
            </div>

            {/* Source Name */}
            <div className="space-y-2">
              <Label htmlFor="source_name">
                Source Name (Optional)
              </Label>
              <Input
                id="source_name"
                type="text"
                placeholder="anthropic-official"
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                A friendly name for this source. Auto-generated if left empty.
              </p>
            </div>

            {/* Branch */}
            <div className="space-y-2">
              <Label htmlFor="branch">Git Branch</Label>
              <Select value={branch} onValueChange={setBranch} disabled={loading}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="main">main</SelectItem>
                  <SelectItem value="master">master</SelectItem>
                  <SelectItem value="develop">develop</SelectItem>
                  <SelectItem value="production">production</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Auto-update */}
            <div className="flex items-center justify-between space-x-2">
              <div className="space-y-0.5">
                <Label htmlFor="auto_update" className="text-base">
                  Enable Auto-Update
                </Label>
                <p className="text-sm text-muted-foreground">
                  Automatically pull updates from Git repository daily
                </p>
              </div>
              <Switch
                id="auto_update"
                checked={autoUpdate}
                onCheckedChange={setAutoUpdate}
                disabled={loading}
              />
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={handleClose}
                disabled={loading}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={loading || !gitUrl}>
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loading ? 'Importing...' : 'Import Repository'}
              </Button>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}
