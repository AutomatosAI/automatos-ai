'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { TeamManagement } from '@/components/team/team-management'
import { usePageAPI } from '@/hooks/use-page-api'

export default function TeamPage() {
    // Use mock config if needed, though we generally want real API for team management
    usePageAPI('team')

    return (
        <MainLayout>
            <TeamManagement />
        </MainLayout>
    )
}
