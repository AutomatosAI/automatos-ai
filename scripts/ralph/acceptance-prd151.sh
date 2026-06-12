#!/bin/bash
# Acceptance gate — PRD-151 Storage Decoupling (MinIO default).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# Verifier fixes baked in: robust CI-minio grep; python3 everywhere.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "factory-only-s3-clients (zero boto3.client outside factory/vectors-backend/bedrock/ops-script)" \
  'cd orchestrator && ! grep -rn "boto3\.client" --include="*.py" . | grep -vE "__pycache__|/tests/|core/storage/s3\.py|backends/s3_vectors_backend\.py|core/llm/clients/bedrock_client\.py|scripts/recreate_s3_index\.py" | grep -q .'

check "fallbacks-deleted (LocalStorageService/LocalImageStore/_local_dir + dead knobs gone)" \
  'cd orchestrator && ! grep -rnE "LocalStorageService|LocalImageStore|_local_dir|MARKETPLACE_LOCAL_DIR|IMAGE_STORE_LOCAL_DIR" --include="*.py" . | grep -v __pycache__ | grep -q .'

check "storage-unit-suites (all new storage suites green offline)" \
  'cd orchestrator && python3 -m pytest tests/test_storage_factory.py tests/test_storage_repoint_api.py tests/test_storage_repoint_modules.py tests/test_storage_repoint_services.py tests/test_storage_vectors_gate.py tests/test_storage_no_fallbacks.py tests/test_storage_presign_lifecycle.py tests/test_storage_failure_modes.py -q -p no:cacheprovider --timeout=120'

check "protected-baseline-suites (77-test DocumentManager/plugin/config baseline stays green)" \
  'cd orchestrator && python3 -m pytest tests/test_rag_ingest_atomicity.py tests/test_plugin_runtime_integration.py tests/test_config.py tests/test_config_env_centralization.py -q -p no:cacheprovider --timeout=120'

check "minio-lane-collectible (skips cleanly without S3_ENDPOINT_URL; runs in CI lane)" \
  'cd orchestrator && python3 -m pytest tests/test_storage_minio_integration.py -q -p no:cacheprovider --timeout=120'

check "import-purity-and-config-hygiene (no import-time I/O; env only via config.py)" \
  'cd orchestrator && python3 -c "import core.storage.s3; import core.services.marketplace_s3; import core.services.image_store; import modules.attachments.store" && ! grep -rn "os.getenv\|os.environ" core/storage/ | grep -q .'

check "ci-workflow-valid (MinIO lane defined, storage env present, YAML parses)" \
  'grep -q "minio" .github/workflows/test.yml && grep -q "server /data" .github/workflows/test.yml && grep -q "S3_ENDPOINT_URL" .github/workflows/test.yml && python3 -c "import yaml; yaml.safe_load(open(\".github/workflows/test.yml\"))"'

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-151 PASS"; else echo "ACCEPTANCE: PRD-151 FAIL"; fi
exit $FAIL
