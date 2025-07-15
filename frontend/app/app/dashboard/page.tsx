
import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import DashboardClient from '@/components/dashboard-client';

export default async function DashboardPage() {
  const session = await getServerSession();

  if (!session) {
    redirect('/login');
  }

  return <DashboardClient />;
}
