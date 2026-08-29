'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { TeamManagement } from '@/components/team/team-management'
import { SaasOnlyNotice } from '@/components/local/saas-only-notice'
import { isRouteAvailableInEdition } from '@/lib/auth-edition'
import { usePageAPI } from '@/hooks/use-page-api'

export default function TeamPage() {
    // Use mock config if needed, though we generally want real API for team management
    usePageAPI('team')

    return (
        <MainLayout>
            {/* PRD-233 S7: Team is a hosted-edition surface — local renders the notice. */}
            {isRouteAvailableInEdition('/team') ? <TeamManagement /> : <SaasOnlyNotice surface="Team" />}
        </MainLayout>
    )
}
