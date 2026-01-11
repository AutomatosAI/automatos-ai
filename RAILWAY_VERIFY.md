# Railway Build Context Verification

## Current Issue
Railway can't find `/orchestrator` and `/frontend` directories even though Root Directory is set.

## What Railway Should See

If **Root Directory = `automatos-ai`**, Railway's build context becomes the `automatos-ai/` directory, so Railway should see:
```
automatos-ai/          (build context root)
├── orchestrator/
│   ├── Dockerfile     (Dockerfile path: orchestrator/Dockerfile)
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── Dockerfile     (Dockerfile path: frontend/Dockerfile)
│   ├── package.json
│   └── ...
└── docker-entrypoint.sh
```

## Verify Railway Settings

### Backend Service:
1. **Root Directory**: Should be exactly `automatos-ai` (case-sensitive)
2. **Dockerfile Path**: Should be `orchestrator/Dockerfile` (relative to Root Directory)
3. **Verify**: The `orchestrator/` directory exists in your GitHub repo at that path

### Frontend Service:
1. **Root Directory**: Should be exactly `automatos-ai` (case-sensitive)
2. **Dockerfile Path**: Should be `frontend/Dockerfile` (relative to Root Directory)
3. **Verify**: The `frontend/` directory exists in your GitHub repo at that path

## Common Issues

1. **Root Directory typo**: Check for typos, extra spaces, or wrong case
2. **Dockerfile Path wrong**: Should be relative to Root Directory
3. **Files not committed**: Make sure `orchestrator/` and `frontend/` are committed to GitHub
4. **Wrong branch**: Railway might be building from wrong branch/commit

## Quick Test

Check if Railway can see your files:
1. Go to Railway → Service → Settings
2. Look for any file browser or "View Files" option
3. Verify you can see `orchestrator/` and `frontend/` directories

If Railway can't see these directories, the Root Directory path is wrong.
