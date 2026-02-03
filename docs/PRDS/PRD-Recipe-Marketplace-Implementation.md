# Implementation Plan: Recipes Marketplace

## Overview
Migrate Workflow Recipes to use the same single-table marketplace architecture as Agents. Currently, recipes exist in `workflow_templates` table but lack marketplace fields (owner_type, workspace_id, cloning tracking).

## Current State
- ✅ Database model exists (`WorkflowTemplate` class)
- ✅ API endpoints exist (`/api/workflow-recipes`)
- ✅ Frontend components exist (recipes-tab, marketplace-recipes-tab)
- ❌ **MISSING**: workspace_id, owner_type, marketplace fields
- ❌ **MISSING**: Share to Marketplace functionality
- ❌ **MISSING**: Install from Marketplace functionality
- ⚠️ **BUG**: API already filters by workspace_id (line 52 of workflow_recipes.py) but field doesn't exist in model!

## Implementation Steps

### 1. Database Migration
**File**: `orchestrator/alembic/versions/20260201_add_marketplace_to_recipes.py`

Add marketplace fields to `workflow_templates` table:
- `workspace_id` (UUID, FK to workspaces.id, nullable) - NULL for marketplace, user's workspace for workspace recipes
- `owner_type` (VARCHAR(20), default='workspace') - 'workspace' or 'marketplace'
- `owner_id` (VARCHAR(255)) - workspace_id for workspace recipes, 'marketplace' for marketplace recipes
- `cloned_from_id` (Integer, FK to workflow_templates.id) - Tracks clone source
- `original_creator_id` (Integer, FK to users.id) - Original creator
- `created_by_user_id` (Integer, FK to users.id) - User who created this instance
- `install_count` (Integer, default=0) - Marketplace install counter
- `is_approved` (Boolean, default=false) - Approval status for marketplace items
- `marketplace_category` (VARCHAR(100)) - Marketplace-specific category
- `marketplace_icon` (VARCHAR(500)) - Icon URL for marketplace
- `recipe_type` (VARCHAR(20), default='complex') - 'simple' or 'complex'

**Indexes**:
- `idx_recipes_owner_type` on (owner_type, is_approved)
- `idx_recipes_marketplace_featured` on (is_featured, install_count) WHERE owner_type='marketplace'
- `idx_recipes_marketplace_category` on (marketplace_category) WHERE owner_type='marketplace'
- `idx_recipes_workspace` on (workspace_id, owner_type)

**Data Migration**:
```sql
UPDATE workflow_templates
SET
    workspace_id = (SELECT id FROM workspaces LIMIT 1),
    owner_id = (SELECT id::varchar FROM workspaces LIMIT 1),
    owner_type = 'workspace',
    is_approved = true,
    recipe_type = 'complex'
WHERE workspace_id IS NULL;
```

### 2. Update Model
**File**: `orchestrator/core/models/core.py` (lines 1026-1109)

Add to WorkflowTemplate class:
- All new field definitions
- Relationships: `cloned_from`, `original_creator`, `created_by_user`
- Property: `is_marketplace_item` returns `self.owner_type == 'marketplace'`
- Update `to_dict()` to include new fields

### 3. Update API Endpoints
**File**: `orchestrator/api/workflow_recipes.py`

**Update existing endpoints**:
- `list_workflow_recipes` (line 25): Change filter to `owner_type='workspace'` AND `workspace_id=ctx.workspace_id`
- `get_workflow_recipe` (line ~107): Add owner_type filter
- `create_workflow_recipe` (line ~132): Set owner_type='workspace', workspace_id, owner_id
- All other endpoints: Add owner_type filtering

**Add new marketplace endpoints**:

```python
@router.post("/submit")
async def submit_recipe_to_marketplace(...)
    # 1. Get workspace recipe
    # 2. Check if user is trusted (5+ approved marketplace items)
    # 3. Clone to marketplace (owner_type='marketplace', workspace_id=None, owner_id='marketplace')
    # 4. Set is_approved based on trusted status
    # 5. Track cloned_from_id and original_creator_id
    # 6. Return success with auto_approved flag

@router.post("/install/{recipe_id}")
async def install_recipe_from_marketplace(...)
    # 1. Get marketplace recipe (owner_type='marketplace', is_approved=True)
    # 2. Clone to workspace (owner_type='workspace', workspace_id=ctx.workspace_id)
    # 3. Handle name collisions (add " (Copy)" suffix)
    # 4. Increment marketplace recipe's install_count
    # 5. Record in marketplace_installs table
    # 6. Auto-clone referenced agents if available in marketplace
    # 7. Return cloned_items and warnings for missing dependencies
```

**Update marketplace.py**:
- Update `list_items` endpoint to query recipes where owner_type='marketplace'
- Return recipe items with type='recipe'

### 4. Frontend Changes

**File**: `frontend/components/workflows/recipes-tab.tsx`

Add "Share to Marketplace" button:
```typescript
const submitMutation = useMutation({
  mutationFn: (params) => apiClient.post('/api/workflow-recipes/submit', params),
  onSuccess: (data) => toast.success(data.message)
})

// Add button to recipe details/card:
<Button onClick={() => submitMutation.mutate({
  recipe_id: recipe.template_id,
  category: recipe.category,
  icon: recipe.icon
})}>
  <Share2 /> Share to Marketplace
</Button>
```

**File**: `frontend/components/marketplace/marketplace-recipes-tab.tsx`

Add Install functionality:
```typescript
const installMutation = useMutation({
  mutationFn: (recipeId) => apiClient.post(`/api/workflow-recipes/install/${recipeId}`),
  onSuccess: (data) => {
    toast.success(data.message)
    data.warnings?.forEach(w => toast.warning(w))
  }
})

// Update Install button:
<Button onClick={() => installMutation.mutate(recipe.template_id)}>
  <Download /> Install
</Button>
```

**File**: `frontend/lib/api-client.ts`

Add methods:
```typescript
submitRecipeToMarketplace(params: {...}): Promise<any>
installRecipeFromMarketplace(recipeId: string): Promise<any>
```

**File**: `frontend/hooks/use-recipe-api.ts`

Add hooks:
```typescript
useSubmitRecipeToMarketplace()
useInstallRecipeFromMarketplace()
```

### 5. Testing Strategy

**Backend Tests**:
- Create workspace recipe with owner_type='workspace'
- Submit to marketplace (trusted user → is_approved=true)
- Submit to marketplace (untrusted user → is_approved=false)
- Install from marketplace (verify clone created in workspace)
- Install with dependencies (verify auto-clone)
- Name collision handling (verify " (Copy)" suffix)

**Frontend Tests**:
- Share button calls submit mutation
- Install button calls install mutation
- Success/error toasts display correctly
- Warnings for missing dependencies display

**Manual Testing**:
- [ ] Create recipe in workspace
- [ ] Submit to marketplace
- [ ] Browse marketplace recipes
- [ ] Install marketplace recipe
- [ ] Verify install_count increments
- [ ] Test with recipe dependencies

## Critical Files

### Backend
- `orchestrator/alembic/versions/20260201_add_marketplace_to_recipes.py` - NEW migration
- `orchestrator/core/models/core.py` (lines 1026-1109) - Update WorkflowTemplate model
- `orchestrator/api/workflow_recipes.py` - Add submit/install endpoints, update existing
- `orchestrator/api/marketplace.py` (line 97+) - Add recipe support to list_items

### Frontend
- `frontend/components/workflows/recipes-tab.tsx` - Add Share button
- `frontend/components/marketplace/marketplace-recipes-tab.tsx` - Add Install button
- `frontend/lib/api-client.ts` - Add submit/install methods
- `frontend/hooks/use-recipe-api.ts` - Add marketplace hooks

## Clone-with-workspace_id Pattern

**Workspace Recipe** (user-created):
```python
owner_type='workspace'
owner_id=str(workspace_id)
workspace_id=user_workspace_id
```

**Marketplace Recipe** (shared):
```python
owner_type='marketplace'
owner_id='marketplace'
workspace_id=None  # No workspace - global item
```

**Installed Recipe** (cloned from marketplace):
```python
owner_type='workspace'
owner_id=str(workspace_id)
workspace_id=user_workspace_id
cloned_from_id=marketplace_recipe.id
original_creator_id=marketplace_recipe.original_creator_id
```

## Verification

After implementation:
1. Create a workspace recipe
2. Submit to marketplace (verify in DB: owner_type='marketplace', workspace_id=NULL)
3. Install from marketplace (verify clone created with owner_type='workspace', workspace_id set)
4. Check install_count incremented on marketplace recipe
5. Verify marketplace_installs table has record
6. Test dependency auto-clone (recipe with agent dependencies)
7. Test name collision handling

## Deployment Steps

1. Backup database
2. Run migration: `alembic upgrade head`
3. Verify indexes created
4. Deploy backend code
5. Deploy frontend code
6. Verify existing recipes still work
7. Test marketplace flow end-to-end
8. Monitor logs for errors
