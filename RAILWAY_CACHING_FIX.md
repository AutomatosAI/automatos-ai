# Railway Caching & Build Optimization

## Current Issues
1. **No pip caching** - Requirements reinstalling every deploy
2. **libmagic missing** - `python-magic` can't find libmagic library
3. **Using mise instead of nixpkgs** - Slower builds

## Solutions

### Option 1: Switch to Railpack (RECOMMENDED)
Railway recommends **Railpack** over Nixpacks for:
- **77% smaller builds**
- **Better caching** (more cache hits)
- **Faster deployments**

**To enable Railpack:**
1. Go to Railway dashboard → Your Backend Service → Settings
2. Under "Build Settings", change builder from "Nixpacks" to **"Railpack"**
3. Railpack auto-detects Python from `requirements.txt` and `Procfile`
4. No config file needed - it just works!

**Railpack automatically:**
- Caches pip packages between builds
- Only rebuilds when `requirements.txt` changes
- Handles system dependencies automatically

### Option 2: Fix Nixpacks (Current Setup)
If staying with Nixpacks, the fixes are in `nixpacks.toml`:

**Caching:**
- Removed `--no-cache-dir` (already done)
- Railway caches based on file hashes
- Ensure `requirements.txt` is stable (no random changes)

**libmagic Fix:**
- Added `LD_LIBRARY_PATH` export in install phase
- `file` package provides libmagic, but needs to be in library path

**Python Provider:**
- Set `[providers] python = "nixpkgs"` to avoid mise
- Deleted `.python-version` file

## Why Caching Isn't Working

Railway caches build layers, but:
1. **If `requirements.txt` hash changes** → Cache invalidated
2. **If build context changes** → Cache invalidated  
3. **Nixpacks limitations** → Less efficient than Railpack

**Check if caching is working:**
- Look for "Using cached" messages in build logs
- If you see "Collecting..." for all packages → Cache not working
- If you see "Using cached" → Cache working!

## Recommended Action

**Switch to Railpack** - It's what Railway recommends and provides:
- Automatic pip caching
- Better build performance
- Smaller images
- No configuration needed

Just change the builder in Railway dashboard settings!
