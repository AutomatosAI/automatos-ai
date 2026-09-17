'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { Bot } from 'lucide-react';
import { useIconStyle } from '@/hooks/use-system-config-api';

export interface PremiumIconProps {
  name: string | null;
  size?: number;
  className?: string;
  /** Override the global style for this instance */
  style?: string;
}

function resolveIconSrc(iconFilename: string, iconStyle: string): string {
  // Defensive: if iconStyle is anything but a non-empty string, fall back
  // to the default path. Prevents "[object Object]" landing in the URL.
  if (typeof iconStyle !== 'string' || iconStyle === '' || iconStyle === 'default') {
    return `/assets/icons/${iconFilename}`;
  }
  return `/assets/icons/${iconStyle}/${iconFilename}`;
}

export function PremiumIcon({ name, size = 24, className = '', style }: PremiumIconProps) {
  const { data: globalStyle } = useIconStyle();
  const [fallback, setFallback] = useState(false);

  // Reset fallback when name or style changes
  const activeStyle = style ?? globalStyle ?? 'default';
  useEffect(() => { setFallback(false); }, [name, activeStyle]);

  if (!name) {
    return <Bot size={size} className={className} />;
  }

  const iconFilename = name.endsWith('.svg') ? name : `${name}.svg`;

  // If the styled icon fails to load, fall back to the default
  const src = fallback
    ? `/assets/icons/${iconFilename}`
    : resolveIconSrc(iconFilename, activeStyle);

  return (
    <Image
      src={src}
      alt={name.replace('.svg', '').replace(/-/g, ' ')}
      width={size}
      height={size}
      className={`object-contain ${className}`}
      unoptimized={true}
      onError={() => {
        if (!fallback) setFallback(true);
      }}
    />
  );
}
