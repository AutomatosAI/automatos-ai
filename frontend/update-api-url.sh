#!/bin/bash
sed -i "s|this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.automatos.app'|this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://206.81.0.227:8000'|" /root/automatos-ai/frontend/lib/api-client.ts
