/**
 * Skill Detail Modal
 * ==================
 * 
 * Displays detailed information about a skill including
 * full content, files, version history, and usage stats.
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Loader2, FileText, Code, Package, GitBranch } from 'lucide-react';
import { useSkillsApi } from '@/hooks/use-skills-api';
import { Skill, SkillContentResponse } from '@/types/skills';
import ReactMarkdown from 'react-markdown';

interface SkillDetailModalProps {
  skill: Skill;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SkillDetailModal({ skill, open, onOpenChange }: SkillDetailModalProps) {
  const { getSkillContent, loading } = useSkillsApi();
  const [content, setContent] = useState<SkillContentResponse | null>(null);

  useEffect(() => {
    if (open && skill) {
      loadContent();
    }
  }, [open, skill]);

  const loadContent = async () => {
    const data = await getSkillContent(skill.id, 2); // Load Level 2: Core
    setContent(data);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <DialogTitle className="text-2xl">{skill.name}</DialogTitle>
              <DialogDescription className="mt-2">
                {skill.description}
              </DialogDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant="secondary">{skill.category}</Badge>
              {skill.skill_version && (
                <Badge variant="outline">v{skill.skill_version}</Badge>
              )}
            </div>
          </div>
        </DialogHeader>

        <Tabs defaultValue="content" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="content">
              <FileText className="mr-2 h-4 w-4" />
              Content
            </TabsTrigger>
            <TabsTrigger value="files">
              <Code className="mr-2 h-4 w-4" />
              Files
            </TabsTrigger>
            <TabsTrigger value="metadata">
              <Package className="mr-2 h-4 w-4" />
              Metadata
            </TabsTrigger>
          </TabsList>

          {/* Content Tab */}
          <TabsContent value="content" className="space-y-4">
            <ScrollArea
              className="h-[500px] w-full rounded-md border p-4 break-words"
              style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}
            >
            {loading ? (
                <div className="flex justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin" />
                </div>
              ) : content ? (
              <div
                className="prose prose-sm max-w-none dark:prose-invert break-words whitespace-pre-wrap overflow-x-hidden prose-pre:whitespace-pre-wrap prose-pre:break-words prose-code:break-words"
                style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}
              >
                  <ReactMarkdown
                    components={{
                      p: ({ node, ...props }) => (
                        <p
                          style={{ wordBreak: 'break-all', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}
                          {...props}
                        />
                      ),
                      li: ({ node, ...props }) => (
                        <li
                          style={{ wordBreak: 'break-all', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}
                          {...props}
                        />
                      ),
                      h1: ({ node, ...props }) => (
                        <h1 style={{ wordBreak: 'break-all', overflowWrap: 'anywhere' }} {...props} />
                      ),
                      h2: ({ node, ...props }) => (
                        <h2 style={{ wordBreak: 'break-all', overflowWrap: 'anywhere' }} {...props} />
                      ),
                      h3: ({ node, ...props }) => (
                        <h3 style={{ wordBreak: 'break-all', overflowWrap: 'anywhere' }} {...props} />
                      ),
                      h4: ({ node, ...props }) => (
                        <h4 style={{ wordBreak: 'break-all', overflowWrap: 'anywhere' }} {...props} />
                      ),
                      pre: ({ node, ...props }) => (
                        <pre
                          style={{ wordBreak: 'break-all', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}
                          {...props}
                        />
                      ),
                      code: ({ node, ...props }) => (
                        <code
                          style={{ wordBreak: 'break-all', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}
                          {...props}
                        />
                      ),
                      td: ({ node, ...props }) => (
                        <td style={{ wordBreak: 'break-all', overflowWrap: 'anywhere' }} {...props} />
                      ),
                      th: ({ node, ...props }) => (
                        <th style={{ wordBreak: 'break-all', overflowWrap: 'anywhere' }} {...props} />
                      ),
                    }}
                  >
                    {content.content}
                  </ReactMarkdown>
                </div>
              ) : (
                <p className="text-muted-foreground">No content available</p>
              )}
            </ScrollArea>
            
            {content && (
              <div className="text-xs text-muted-foreground">
                Estimated tokens: {content.estimated_tokens}
              </div>
            )}
          </TabsContent>

          {/* Files Tab */}
          <TabsContent value="files" className="space-y-4">
            <ScrollArea className="h-[500px] w-full rounded-md border p-4">
              {skill.files && skill.files.length > 0 ? (
                <div className="space-y-2">
                  {skill.files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 rounded-lg border bg-card"
                    >
                      <div className="flex-1">
                        <p className="font-medium text-sm">{file.file_path}</p>
                        {file.content_summary && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {file.content_summary}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <Badge variant="outline">{file.file_type}</Badge>
                        {file.estimated_tokens && (
                          <span>{file.estimated_tokens} tokens</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground">No files indexed</p>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Metadata Tab */}
          <TabsContent value="metadata" className="space-y-4">
            <ScrollArea className="h-[500px] w-full rounded-md border p-4">
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">General</h4>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Type:</dt>
                      <dd>{skill.skill_type}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Category:</dt>
                      <dd>{skill.category}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Version:</dt>
                      <dd>{skill.skill_version || 'N/A'}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Source:</dt>
                      <dd>{skill.skill_source || 'Unknown'}</dd>
                    </div>
                  </dl>
                </div>

                {skill.tags.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2">Tags</h4>
                    <div className="flex flex-wrap gap-2">
                      {skill.tags.map(tag => (
                        <Badge key={tag} variant="secondary">{tag}</Badge>
                      ))}
                    </div>
                  </div>
                )}

                {skill.git_repo_url && (
                  <div>
                    <h4 className="font-semibold mb-2 flex items-center">
                      <GitBranch className="mr-2 h-4 w-4" />
                      Git Information
                    </h4>
                    <dl className="space-y-2 text-sm">
                      <div>
                        <dt className="text-muted-foreground mb-1">Repository:</dt>
                        <dd className="break-all text-xs">{skill.git_repo_url}</dd>
                      </div>
                      {skill.git_commit_sha && (
                        <div>
                          <dt className="text-muted-foreground mb-1">Commit:</dt>
                          <dd className="font-mono text-xs">
                            {skill.git_commit_sha.substring(0, 8)}
                          </dd>
                        </div>
                      )}
                    </dl>
                  </div>
                )}

                <div>
                  <h4 className="font-semibold mb-2">Timestamps</h4>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Created:</dt>
                      <dd>{new Date(skill.created_at).toLocaleDateString()}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Updated:</dt>
                      <dd>{new Date(skill.updated_at).toLocaleDateString()}</dd>
                    </div>
                    {skill.last_sync_at && (
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Last Sync:</dt>
                        <dd>{new Date(skill.last_sync_at).toLocaleDateString()}</dd>
                      </div>
                    )}
                  </dl>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
