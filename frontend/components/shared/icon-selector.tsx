'use client';

import React, { useState, useMemo } from 'react';
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from '@/components/ui/popover';
import { Button } from '@/components/ui/button';
import { SearchInput, PremiumIcon } from '@/components/shared';
import { Settings2, X } from 'lucide-react';
import iconRegistry from '@/config/iconRegistry.json';

export interface IconSelectorProps {
    value?: string | null;
    onChange: (iconName: string | null) => void;
    triggerLabel?: string;
    triggerClassName?: string;
}

export function IconSelector({
    value,
    onChange,
    triggerLabel = 'Select Icon',
    triggerClassName = '',
}: IconSelectorProps) {
    const [open, setOpen] = useState(false);
    const [search, setSearch] = useState('');

    // Filter icons based on search query against name and tags
    const filteredIcons = useMemo(() => {
        if (!search.trim()) return iconRegistry;

        const query = search.toLowerCase();
        return iconRegistry.filter((icon: any) => {
            if (icon.name.toLowerCase().includes(query)) return true;
            if (icon.tags && icon.tags.some((tag: string) => tag.includes(query))) return true;
            return false;
        });
    }, [search]);

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button
                    variant="outline"
                    className={`h-12 justify-between px-4 ${triggerClassName}`}
                >
                    <div className="flex items-center gap-3">
                        {value ? (
                            <PremiumIcon name={value} size={24} />
                        ) : (
                            <div className="w-6 h-6 rounded-md bg-secondary/50 flex items-center justify-center shrink-0">
                                <Settings2 className="w-4 h-4 text-muted-foreground" />
                            </div>
                        )}
                        <span className={value ? 'text-foreground' : 'text-muted-foreground font-normal'}>
                            {value ? iconRegistry.find((i: any) => i.id === value)?.name || value : triggerLabel}
                        </span>
                    </div>
                </Button>
            </PopoverTrigger>

            <PopoverContent className="w-[340px] p-0 glass-card rounded-2xl overflow-hidden shadow-2xl" align="start">

                {/* Header / Search */}
                <div className="p-3 border-b border-border/50 bg-background/50 backdrop-blur-md sticky top-0 z-10">
                    <div className="flex items-center justify-between mb-3 px-1">
                        <span className="text-sm font-semibold">Choose an Icon</span>
                        {value && (
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onChange(null);
                                    setOpen(false);
                                }}
                                className="h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
                            >
                                Clear
                            </Button>
                        )}
                    </div>
                    <SearchInput
                        value={search}
                        onChange={setSearch}
                        placeholder="Search icons (e.g. brain, marketing)..."
                        className="w-full"
                    />
                </div>

                {/* Scrollable Icon Grid */}
                <div className="p-3 bg-card/80 max-h-[300px] overflow-y-auto overflow-x-hidden">
                    {filteredIcons.length === 0 ? (
                        <div className="py-8 text-center text-sm text-muted-foreground">
                            No icons found for "{search}"
                        </div>
                    ) : (
                        <div className="grid grid-cols-4 gap-2">
                            {filteredIcons.slice(0, 100).map((icon: any) => (
                                <button
                                    key={icon.id}
                                    onClick={() => {
                                        onChange(icon.id);
                                        setOpen(false);
                                    }}
                                    className={`
                    flex flex-col items-center justify-center gap-2 p-2 rounded-xl transition-all
                    hover:bg-primary/10 hover:scale-105 active:scale-95
                    ${value === icon.id ? 'bg-primary/20 ring-1 ring-primary' : ''}
                  `}
                                    title={icon.name}
                                >
                                    <PremiumIcon name={icon.id} size={32} />
                                </button>
                            ))}
                        </div>
                    )}
                    {filteredIcons.length > 100 && (
                        <div className="mt-3 text-xs text-center text-muted-foreground pt-2 border-t border-border/30">
                            Showing top 100 results. Use search to find more.
                        </div>
                    )}
                </div>
            </PopoverContent>
        </Popover>
    );
}
