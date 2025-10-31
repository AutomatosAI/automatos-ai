/**
 * Skill Browser Component
 * =======================
 * 
 * Main interface for browsing and searching skills.
 * 
 * Features:
 * - Search and filter skills
 * - Category filtering
 * - Source filtering
 * - Tag filtering
 * - Grid/list view
 * - Skill cards with preview
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Search, Grid, List, Filter, Tag, Package } from 'lucide-react';
import { useSkillsApi } from '@/hooks/use-skills-api';
import { Skill } from '@/types/skills';
import { SkillDetailModal } from './skill-detail-modal';

interface SkillBrowserProps {
  onSkillSelect?: (skill: Skill) => void;
  selectionMode?: boolean;
  selectedSkillIds?: number[];
  hideUnknownSource?: boolean;
}

export function SkillBrowser({ 
  onSkillSelect, 
  selectionMode = false,
  selectedSkillIds = [],
  hideUnknownSource = false,
}: SkillBrowserProps) {
  const { listSkills, loading } = useSkillsApi();
  
  const [skills, setSkills] = useState<Skill[]>([]);
  const [filteredSkills, setFilteredSkills] = useState<Skill[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [selectedSkill, setSelectedSkill] = useState<Skill | null>(null);

  // Load skills on mount
  useEffect(() => {
    loadSkills();
  }, []);

  // Apply filters
  useEffect(() => {
    let filtered = skills;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(skill =>
        skill.name.toLowerCase().includes(query) ||
        skill.description?.toLowerCase().includes(query) ||
        skill.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Category filter
    if (categoryFilter && categoryFilter !== 'all') {
      filtered = filtered.filter(skill => skill.category === categoryFilter);
    }

    // Source filter
    if (sourceFilter && sourceFilter !== 'all') {
      filtered = filtered.filter(skill => skill.skill_source === sourceFilter);
    }

    if (hideUnknownSource) {
      filtered = filtered.filter(skill => !!skill.skill_source && skill.skill_source !== 'Unknown');
    }

    setFilteredSkills(filtered);
  }, [skills, searchQuery, categoryFilter, sourceFilter]);

  const loadSkills = async () => {
    const data = await listSkills({ limit: 200 });
    setSkills(data);
    setFilteredSkills(data);
  };

  const handleSkillClick = (skill: Skill) => {
    if (selectionMode && onSkillSelect) {
      onSkillSelect(skill);
    } else {
      setSelectedSkill(skill);
    }
  };

  // Extract unique categories and sources
  const categories = Array.from(new Set(skills.map(s => s.category)));
  const sources = Array.from(new Set(skills.map(s => s.skill_source).filter(Boolean)));

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Search */}
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search skills by name, description, or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Category Filter */}
        <Select value={categoryFilter} onValueChange={setCategoryFilter}>
          <SelectTrigger className="w-[180px]">
            <Filter className="mr-2 h-4 w-4" />
            <SelectValue placeholder="Category" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {categories.map(cat => (
              <SelectItem key={cat} value={cat}>
                {cat}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Source Filter */}
        <Select value={sourceFilter} onValueChange={setSourceFilter}>
          <SelectTrigger className="w-[180px]">
            <Package className="mr-2 h-4 w-4" />
            <SelectValue placeholder="Source" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Sources</SelectItem>
            {sources.map(src => (
              <SelectItem key={src} value={src!}>
                {src}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* View Toggle */}
        <div className="flex gap-2">
          <Button
            variant={viewMode === 'grid' ? 'default' : 'outline'}
            size="icon"
            onClick={() => setViewMode('grid')}
          >
            <Grid className="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="icon"
            onClick={() => setViewMode('list')}
          >
            <List className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Results Count */}
      <div className="text-sm text-muted-foreground">
        Showing {filteredSkills.length} of {skills.length} skills
      </div>

      {/* Skills Grid/List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent" />
          <p className="mt-4 text-muted-foreground">Loading skills...</p>
        </div>
      ) : filteredSkills.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No skills found</p>
        </div>
      ) : (
        <div className={
          viewMode === 'grid'
            ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
            : 'space-y-4'
        }>
          {filteredSkills.map(skill => (
            <SkillCard
              key={skill.id}
              skill={skill}
              onClick={() => handleSkillClick(skill)}
              selected={selectedSkillIds.includes(skill.id)}
              compact={viewMode === 'list'}
            />
          ))}
        </div>
      )}

      {/* Skill Detail Modal */}
      {selectedSkill && (
        <SkillDetailModal
          skill={selectedSkill}
          open={!!selectedSkill}
          onOpenChange={(open) => !open && setSelectedSkill(null)}
        />
      )}
    </div>
  );
}

// Skill Card Component
interface SkillCardProps {
  skill: Skill;
  onClick: () => void;
  selected?: boolean;
  compact?: boolean;
}

function SkillCard({ skill, onClick, selected, compact }: SkillCardProps) {
  return (
    <Card
      className={`cursor-pointer transition-all hover:shadow-lg ${
        selected ? 'ring-2 ring-primary' : ''
      } ${compact ? 'flex items-start' : ''}`}
      onClick={onClick}
    >
      <CardHeader className={compact ? 'pb-2' : ''}>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-lg">{skill.name}</CardTitle>
            {skill.skill_version && (
              <p className="text-xs text-muted-foreground mt-1">
                v{skill.skill_version}
              </p>
            )}
          </div>
          <Badge variant="secondary">{skill.category}</Badge>
        </div>
        <CardDescription className="line-clamp-2">
          {skill.description || 'No description available'}
        </CardDescription>
      </CardHeader>
      
      {!compact && (
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {skill.tags.slice(0, 3).map(tag => (
              <Badge key={tag} variant="outline" className="text-xs">
                <Tag className="mr-1 h-3 w-3" />
                {tag}
              </Badge>
            ))}
            {skill.tags.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{skill.tags.length - 3} more
              </Badge>
            )}
          </div>
        </CardContent>
      )}
      
      <CardFooter className="text-xs text-muted-foreground">
        <div className="flex items-center justify-between w-full">
          <span>Source: {skill.skill_source || 'Unknown'}</span>
          {skill.files && skill.files.length > 0 && (
            <span>{skill.files.length} files</span>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}
