import React from 'react';
import Image from 'next/image';
import { Bot } from 'lucide-react';

export interface PremiumIconProps {
  name: string | null;
  size?: number;
  className?: string;
}

export function PremiumIcon({ name, size = 24, className = '' }: PremiumIconProps) {
  if (!name) {
    return <Bot size={size} className={className} />;
  }

  return (
    <Image
      src={`/assets/icons/${name}`}
      alt={name.replace('.svg', '').replace(/-/g, ' ')}
      width={size}
      height={size}
      className={`object-contain ${className}`}
      unoptimized={true}
    />
  );
}
