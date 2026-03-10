import React from 'react';
import Image from 'next/image';

interface PremiumIconProps {
  name: string;
  size?: number;
  className?: string;
  alt?: string;
}

export function PremiumIcon({
  name,
  size = 24,
  className = '',
  alt,
}: PremiumIconProps) {
  // Graceful fallback if name is unexpectedly missing
  if (!name) return null;

  const src = `/assets/icons/${name.toLowerCase()}.svg`;
  
  return (
    <div 
      className={`inline-flex items-center justify-center shrink-0 ${className}`}
      style={{ width: size, height: size }}
    >
      <Image
        src={src}
        alt={alt || `${name} icon`}
        width={size}
        height={size}
        className="object-contain"
        unoptimized // SVGs don't need Next.js image optimization
      />
    </div>
  );
}
