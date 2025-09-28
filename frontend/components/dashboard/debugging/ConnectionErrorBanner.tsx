import React from 'react';
import { AlertTriangle, RefreshCcw, Database, Server, Users } from 'lucide-react';
import { Button } from '../../ui/button';
import { useRouter } from 'next/navigation';

interface ConnectionErrorBannerProps {
  failedServices: string[];
  totalServices: number;
  onRetry: () => void;
}

export function ConnectionErrorBanner({ 
  failedServices,
  totalServices,
  onRetry
}: ConnectionErrorBannerProps) {
  const router = useRouter();
  const allFailed = failedServices.length === totalServices;
  
  // Service specific icons
  const getServiceIcon = (service: string) => {
    switch(service) {
      case 'systemHealth': return <Server className='h-4 w-4' />;
      case 'agents': return <Users className='h-4 w-4' />;
      case 'documents': return <Database className='h-4 w-4' />;
      default: return <AlertTriangle className='h-4 w-4' />;
    }
  };

  if (allFailed) {
    return (
      <div className='bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6'>
        <div className='flex items-start'>
          <div className='flex-shrink-0'>
            <AlertTriangle className='h-5 w-5 text-red-400' aria-hidden='true' />
          </div>
          <div className='ml-3 w-full'>
            <h3 className='text-sm font-medium text-red-800 dark:text-red-300'>
              System Connectivity Issues
            </h3>
            <div className='mt-2 text-sm text-red-700 dark:text-red-200'>
              <p>
                Unable to connect to backend services. This could be due to network issues or the server may be down.
              </p>
              
              <ul className='list-disc list-inside mt-2'>
                {failedServices.map(service => (
                  <li key={service} className='flex items-center space-x-1'>
                    {getServiceIcon(service)} <span>{service}</span>
                  </li>
                ))}
              </ul>
              
              <div className='mt-4 flex space-x-3'>
                <Button 
                  variant='outline' 
                  size='sm'
                  onClick={onRetry} 
                  className='inline-flex items-center text-red-700 bg-red-50 hover:bg-red-100 border border-red-200 dark:border-red-800 dark:text-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50'
                >
                  <RefreshCcw className='mr-2 h-4 w-4' />
                  Retry Connection
                </Button>
                
                <Button 
                  variant='outline' 
                  size='sm'
                  onClick={() => router.push('/agents')} 
                  className='inline-flex items-center text-red-700 bg-red-50 hover:bg-red-100 border border-red-200 dark:border-red-800 dark:text-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50'
                >
                  <Users className='mr-2 h-4 w-4' />
                  Go to Agents
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Partial failure banner - less severe
  return (
    <div className='bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4 mb-6'>
      <div className='flex'>
        <div className='flex-shrink-0'>
          <AlertTriangle className='h-5 w-5 text-amber-400' aria-hidden='true' />
        </div>
        <div className='ml-3 flex-1 md:flex md:justify-between'>
          <p className='text-sm text-amber-700 dark:text-amber-200'>
            Some services are experiencing issues ({failedServices.length}/{totalServices} failed)
          </p>
          <div className='mt-3 text-sm md:mt-0 md:ml-6'>
            <Button 
              variant='ghost' 
              size='sm'
              onClick={onRetry} 
              className='whitespace-nowrap font-medium text-amber-700 hover:text-amber-600 dark:text-amber-200 dark:hover:text-amber-100'
            >
              <RefreshCcw className='mr-2 h-4 w-4' />
              Retry
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
