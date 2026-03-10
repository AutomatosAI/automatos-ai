'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { IconSelector } from '@/components/shared';
import { Save, Loader2, Palette } from 'lucide-react';
import { useSystemIcons, useUpdateSystemConfigKey } from '@/hooks/use-system-config-api';
import { useToast } from '@/components/ui/use-toast';

// Updated categories from user screenshots
const DEFAULT_CATEGORIES = [
    // Agent Categories
    { id: 'analytics', label: 'Analytics', type: 'agent' },
    { id: 'business', label: 'Business', type: 'agent' },
    { id: 'communication_agent', label: 'Communication', type: 'agent' },
    { id: 'design', label: 'Design', type: 'agent' },
    { id: 'development', label: 'Development', type: 'agent' },
    { id: 'education', label: 'Education', type: 'agent' },
    { id: 'general', label: 'General', type: 'agent' },
    { id: 'hr', label: 'HR', type: 'agent' },
    { id: 'legal', label: 'Legal', type: 'agent' },
    { id: 'marketing', label: 'Marketing', type: 'agent' },
    { id: 'productivity', label: 'Productivity', type: 'agent' },
    { id: 'research', label: 'Research', type: 'agent' },
    { id: 'sales', label: 'Sales', type: 'agent' },
    { id: 'support', label: 'Support', type: 'agent' },
    { id: 'writing', label: 'Writing', type: 'agent' },
    { id: 'custom', label: 'Custom', type: 'agent' },

    // Skill Categories
    { id: 'cognitive', label: 'Cognitive Skills', type: 'skill' },
    { id: 'technical', label: 'Technical Skills', type: 'skill' },
    { id: 'communication_skill', label: 'Communication Skills', type: 'skill' },
    { id: 'analytical', label: 'Analytical Skills', type: 'skill' },
    { id: 'creative', label: 'Creative Skills', type: 'skill' },
    { id: 'system', label: 'System \u0026 Infrastructure', type: 'skill' },
];

export function SystemIconsSettingsTab() {
    const { toast } = useToast();
    const [mappings, setMappings] = useState<Record<string, string | null>>({});

    const { data: iconMappings, isLoading } = useSystemIcons();
    const updateMappingsMutation = useUpdateSystemConfigKey();

    useEffect(() => {
        if (iconMappings) {
            setMappings(iconMappings);
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

    if (isLoading) {
        return (
            <div className="flex items-center justify-center p-12">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    const agentCategories = DEFAULT_CATEGORIES.filter(c => c.type === 'agent');
    const skillCategories = DEFAULT_CATEGORIES.filter(c => c.type === 'skill');

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">System Icon Mappings</h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        Assign premium SVG icons to platform categories. These will act as the default avatars across the platform.
                    </p>
                </div>
                <Button onClick={handleSave} disabled={updateMappingsMutation.isLoading}>
                    {updateMappingsMutation.isLoading ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                        <Save className="w-4 h-4 mr-2" />
                    )}
                    Save Mappings
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* Agent Categories */}
                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle className="text-lg">Agent Personas \u0026 Categories</CardTitle>
                        <CardDescription>Default icons for agent avatar generation</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {agentCategories.map(cat => (
                            <div key={cat.id} className="flex items-center justify-between p-3 rounded-xl bg-secondary/20 border border-border/30">
                                <span className="font-medium text-sm">{cat.label}</span>
                                <IconSelector
                                    value={mappings[cat.id] || null}
                                    onChange={(val) => handleIconChange(cat.id, val)}
                                    triggerLabel="Assign Icon"
                                    triggerClassName="w-[180px]"
                                />
                            </div>
                        ))}
                    </CardContent>
                </Card>

                {/* Skill Categories */}
                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle className="text-lg">Skill Domains</CardTitle>
                        <CardDescription>Default icons for skill badges and blocks</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {skillCategories.map(cat => (
                            <div key={cat.id} className="flex items-center justify-between p-3 rounded-xl bg-secondary/20 border border-border/30">
                                <span className="font-medium text-sm">{cat.label}</span>
                                <IconSelector
                                    value={mappings[cat.id] || null}
                                    onChange={(val) => handleIconChange(cat.id, val)}
                                    triggerLabel="Assign Icon"
                                    triggerClassName="w-[180px]"
                                />
                            </div>
                        ))}
                    </CardContent>
                </Card>

            </div>
        </div>
    );
}
