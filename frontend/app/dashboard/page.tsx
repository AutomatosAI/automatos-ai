import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import DashboardClient from '@/components/dashboard-client';
import { authOptions } from '@/lib/auth';

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);

    redirect('/login');
  }

  return <DashboardClient />;
}
