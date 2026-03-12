'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { PremiumIcon, IconSelector } from '@/components/shared';
import { Save, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import { useSystemIcons, useIconStyle, useUpdateSystemConfigKey } from '@/hooks/use-system-config-api';
import { useToast } from '@/components/ui/use-toast';
import { cn } from '@/lib/utils';

// ─── Icon Mapping Definitions ────────────────────────────────────────
// Each section groups related mappings. IDs are the keys stored in DB.

interface IconMapping {
    id: string
    label: string
}

interface IconSection {
    title: string
    description: string
    items: IconMapping[]
    defaultOpen?: boolean
}

// ─── Available Icon Styles ───────────────────────────────────────────

interface IconStyleOption {
    id: string
    label: string
    description: string
    color: string
    previewIcons: string[]
}

const ICON_STYLES: IconStyleOption[] = [
    {
        id: 'default',
        label: 'Core Gradient',
        description: 'Default multi-color gradient icons',
        color: '',
        previewIcons: ['cyborg', 'graph', 'dashboard-vertical-rectangle-4'],
    },
    {
        id: 'core-line-orange',
        label: 'Core Line — Orange',
        description: 'Clean line icons in brand orange',
        color: '#F97316',
        previewIcons: ['brain', 'cog', 'graph'],
    },
    {
        id: 'core-line-blue',
        label: 'Core Line — Blue',
        description: 'Clean line icons in blue',
        color: '#3B82F6',
        previewIcons: ['brain', 'cog', 'graph'],
    },
    {
        id: 'core-line-green',
        label: 'Core Line — Green',
        description: 'Clean line icons in green',
        color: '#22C55E',
        previewIcons: ['brain', 'cog', 'graph'],
    },
    {
        id: 'core-line-red',
        label: 'Core Line — Red',
        description: 'Clean line icons in red',
        color: '#EF4444',
        previewIcons: ['brain', 'cog', 'graph'],
    },
    {
        id: 'core-line-yellow',
        label: 'Core Line — Yellow',
        description: 'Clean line icons in yellow',
        color: '#EAB308',
        previewIcons: ['brain', 'cog', 'graph'],
    },
    {
        id: 'core-line-purple',
        label: 'Core Line — Purple',
        description: 'Clean line icons in purple',
        color: '#A855F7',
        previewIcons: ['brain', 'cog', 'graph'],
    },
];

const ICON_SECTIONS: IconSection[] = [
    {
        title: 'Sidebar Navigation',
        description: 'Main menu icons in the sidebar',
        defaultOpen: true,
        items: [
            { id: 'nav_chat', label: 'Chat' },
            { id: 'nav_activity', label: 'Activity' },
            { id: 'nav_agents', label: 'Agents' },
            { id: 'nav_tools', label: 'Tools & Integrations' },
            { id: 'nav_marketplace', label: 'Marketplace' },
            { id: 'nav_knowledge', label: 'Knowledge Base' },
            { id: 'nav_team', label: 'Team' },
            { id: 'nav_context', label: 'Context Engineering' },
            { id: 'nav_dashboard', label: 'Dashboard' },
            { id: 'nav_analytics', label: 'Analytics' },
            { id: 'nav_settings', label: 'Settings' },
        ],
    },
    {
        title: 'Platform Entities',
        description: 'Default icons for core platform objects — used across stats bars, cards, and empty states',
        defaultOpen: true,
        items: [
            { id: 'global_agent', label: 'Agents' },
            { id: 'global_document', label: 'Documents' },
            { id: 'global_tool', label: 'Tools' },
            { id: 'global_skill', label: 'Skills / Capabilities' },
            { id: 'global_plugin', label: 'Plugins / Widgets' },
            { id: 'global_recipe', label: 'Recipes / Workflows' },
            { id: 'global_workflow', label: 'Workflows' },
            { id: 'global_store', label: 'Marketplace' },
            { id: 'global_analytics', label: 'Analytics' },
            { id: 'global_storage', label: 'Storage' },
            { id: 'global_cost', label: 'Cost / Billing' },
            { id: 'global_performance', label: 'Performance' },
            { id: 'global_channel', label: 'Channels / Connections' },
            { id: 'global_activity', label: 'Activity / Status' },
            { id: 'global_featured', label: 'Featured / Trending' },
            { id: 'global_trigger', label: 'Triggers / Automation' },
        ],
    },
    {
        title: 'Agent Categories',
        description: 'Default avatars for each agent persona category',
        defaultOpen: false,
        items: [
            { id: 'analytics', label: 'Analytics' },
            { id: 'business', label: 'Business' },
            { id: 'communication', label: 'Communication' },
            { id: 'design', label: 'Design' },
            { id: 'development', label: 'Development' },
            { id: 'education', label: 'Education' },
            { id: 'general', label: 'General' },
            { id: 'hr', label: 'HR' },
            { id: 'legal', label: 'Legal' },
            { id: 'marketing', label: 'Marketing' },
            { id: 'productivity', label: 'Productivity' },
            { id: 'research', label: 'Research' },
            { id: 'sales', label: 'Sales' },
            { id: 'support', label: 'Support' },
            { id: 'writing', label: 'Writing' },
            { id: 'custom', label: 'Custom' },
        ],
    },
    {
        title: 'Skill Domains',
        description: 'Default icons for skill badges and blocks',
        defaultOpen: false,
        items: [
            { id: 'cognitive', label: 'Cognitive' },
            { id: 'technical', label: 'Technical' },
            { id: 'communication_skill', label: 'Communication' },
            { id: 'analytical', label: 'Analytical' },
            { id: 'creative', label: 'Creative' },
            { id: 'system', label: 'System & Infrastructure' },
        ],
    },
];

// ─── Section Component ───────────────────────────────────────────────

function IconMappingSection({
    section,
    mappings,
    onIconChange,
}: {
    section: IconSection
    mappings: Record<string, string | null>
    onIconChange: (id: string, val: string | null) => void
}) {
    const [open, setOpen] = useState(section.defaultOpen ?? false);
    const assignedCount = section.items.filter(i => mappings[i.id]).length;

    return (
        <div className="border border-border/40 rounded-xl overflow-hidden bg-secondary/5">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center justify-between w-full px-4 py-3 hover:bg-secondary/20 transition-colors text-left"
            >
                <div className="flex items-center gap-3">
                    {open ? (
                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                    ) : (
                        <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    )}
                    <div>
                        <span className="font-semibold text-sm">{section.title}</span>
                        <span className="text-xs text-muted-foreground ml-2">
                            {assignedCount}/{section.items.length} assigned
                        </span>
                    </div>
                </div>
                <p className="text-xs text-muted-foreground hidden sm:block max-w-[300px] text-right">
                    {section.description}
                </p>
            </button>

            {open && (
                <div className="px-4 pb-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                    {section.items.map(item => (
                        <div
                            key={item.id}
                            className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg bg-background/40 border border-border/20"
                        >
                            <div className="flex items-center gap-2 min-w-0">
                                {mappings[item.id] ? (
                                    <PremiumIcon name={mappings[item.id]!} size={18} className="shrink-0" />
                                ) : (
                                    <div className="w-[18px] h-[18px] rounded bg-secondary/40 shrink-0" />
                                )}
                                <span className="text-xs font-medium truncate">{item.label}</span>
                            </div>
                            <IconSelector
                                value={mappings[item.id] || null}
                                onChange={(val) => onIconChange(item.id, val)}
                                triggerLabel=""
                                triggerClassName="w-[120px] h-7 text-xs"
                                compact
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

// ─── Style Picker Component ─────────────────────────────────────────

function IconStylePicker({
    activeStyle,
    onStyleChange,
    isSaving,
}: {
    activeStyle: string
    onStyleChange: (styleId: string) => void
    isSaving: boolean
}) {
    return (
        <div className="border border-border/40 rounded-xl overflow-hidden bg-secondary/5">
            <div className="px-5 py-4 border-b border-border/20">
                <h3 className="font-semibold text-base">Icon Style</h3>
                <p className="text-sm text-muted-foreground mt-1">
                    Switch all platform icons to a different visual style. Changes apply instantly across the entire platform.
                </p>
            </div>
            <div className="p-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {ICON_STYLES.map(style => {
                    const isActive = activeStyle === style.id;
                    return (
                        <button
                            key={style.id}
                            onClick={() => onStyleChange(style.id)}
                            disabled={isSaving}
                            className={cn(
                                'flex flex-col gap-3 p-4 rounded-xl border-2 transition-all text-left',
                                'hover:bg-secondary/20 hover:border-primary/40 hover:shadow-md',
                                isActive
                                    ? 'bg-primary/10 border-primary ring-2 ring-primary/30 shadow-sm'
                                    : 'border-border/30 bg-background/40'
                            )}
                        >
                            {/* Preview icons row */}
                            <div className="flex items-center gap-3 w-full">
                                <div className="flex gap-2 bg-background/60 rounded-lg p-2.5">
                                    {style.previewIcons.map(icon => (
                                        <PremiumIcon key={icon} name={icon} size={28} style={style.id} />
                                    ))}
                                </div>
                                {style.color && (
                                    <div
                                        className="w-4 h-4 rounded-full ml-auto shrink-0 ring-1 ring-border/30"
                                        style={{ backgroundColor: style.color }}
                                    />
                                )}
                            </div>
                            {/* Label + description */}
                            <div className="space-y-0.5">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold">{style.label}</span>
                                    {isActive && (
                                        <span className="text-[10px] font-medium bg-primary/20 text-primary px-1.5 py-0.5 rounded-full">
                                            Active
                                        </span>
                                    )}
                                </div>
                                <p className="text-xs text-muted-foreground leading-relaxed">
                                    {style.description}
                                </p>
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}

// ─── Main Component ──────────────────────────────────────────────────

export function SystemIconsSettingsTab() {
    const { toast } = useToast();
    const [mappings, setMappings] = useState<Record<string, string | null>>({});

    const { data: iconMappings, isLoading } = useSystemIcons();
    const { data: activeStyle, isLoading: styleLoading } = useIconStyle();
    const updateMappingsMutation = useUpdateSystemConfigKey();
    const updateStyleMutation = useUpdateSystemConfigKey();

    useEffect(() => {
        if (iconMappings) {
            // Migrate legacy key: communication_agent → communication
            const migrated = { ...iconMappings };
            if (migrated['communication_agent'] && !migrated['communication']) {
                migrated['communication'] = migrated['communication_agent'];
            }
            delete migrated['communication_agent'];
            setMappings(migrated);
        }
    }, [iconMappings]);

    const handleSave = async () => {
        try {
            await updateMappingsMutation.mutateAsync({
                key: 'system_icon_mappings',
                value: mappings
            });
            toast({
                title: 'Success',
                description: 'System icon mappings saved successfully.',
            });
        } catch (error) {
            console.error('Failed to save mappings', error);
            toast({
                title: 'Error',
                description: 'Failed to save system icon mappings.',
                variant: 'destructive'
            });
        }
    };

    const handleIconChange = (categoryId: string, iconId: string | null) => {
        setMappings(prev => ({
            ...prev,
            [categoryId]: iconId
        }));
    };

    const handleStyleChange = async (styleId: string) => {
        try {
            await updateStyleMutation.mutateAsync({
                key: 'active_icon_style',
                value: styleId,
            });
            toast({
                title: 'Style updated',
                description: `Icons switched to ${ICON_STYLES.find(s => s.id === styleId)?.label ?? styleId}`,
            });
        } catch (error) {
            console.error('Failed to update icon style', error);
            toast({
                title: 'Error',
                description: 'Failed to update icon style.',
                variant: 'destructive',
            });
        }
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center p-12">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    const totalAssigned = Object.values(mappings).filter(Boolean).length;
    const totalSlots = ICON_SECTIONS.reduce((sum, s) => sum + s.items.length, 0);

    return (
        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">System Icon Mappings</h2>
                    <p className="text-sm text-muted-foreground mt-0.5">
                        {totalAssigned}/{totalSlots} icons assigned — updates apply platform-wide
                    </p>
                </div>
                <Button onClick={handleSave} disabled={updateMappingsMutation.isLoading} size="sm">
                    {updateMappingsMutation.isLoading ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                        <Save className="w-4 h-4 mr-2" />
                    )}
                    Save
                </Button>
            </div>

            <IconStylePicker
                activeStyle={activeStyle ?? 'default'}
                onStyleChange={handleStyleChange}
                isSaving={updateStyleMutation.isLoading}
            />

            {ICON_SECTIONS.map(section => (
                <IconMappingSection
                    key={section.title}
                    section={section}
                    mappings={mappings}
                    onIconChange={handleIconChange}
                />
            ))}
        </div>
    );
}
