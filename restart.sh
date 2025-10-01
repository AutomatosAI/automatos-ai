cd /root/automatos-ai/frontend && pkill -f 'npm start' || true && pkill -f 'next' || true && sleep 2 && NODE_ENV=development npm run dev > /root/automatos-ai/frontend.log 2>&1 &
