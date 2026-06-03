#!/usr/bin/env bash
# Deploy MLflow UI as a Databricks App preview.
#
# Usage:
#   bash dev/deploy-ui-preview.sh [--profile PROFILE] [--app-name APP_NAME]
#
# Defaults:
#   --profile   df1
#   --app-name  mlflow-ui-preview-dev

set -euo pipefail

PROFILE="df1"
APP_NAME="mlflow-ui-preview-dev"

while [[ $# -gt 0 ]]; do
  case $1 in
    --profile)  PROFILE="$2";   shift 2 ;;
    --app-name) APP_NAME="$2";  shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"
JS_DIR="$REPO_ROOT/mlflow/server/js"

echo "==> Cleaning previous build artifacts"
rm -rf "$JS_DIR/build" "$DIST_DIR"

echo "==> Building MLflow wheel (without UI assets to stay under 10MB)"
cd "$REPO_ROOT"
uv build --wheel
WHEEL_PATH=$(find "$DIST_DIR" -name "*.whl" | head -1)
WHEEL_NAME=$(basename "$WHEEL_PATH")
WHEEL_SIZE=$(stat -f%z "$WHEEL_PATH" 2>/dev/null || stat -c%s "$WHEEL_PATH")
echo "    Wheel: $WHEEL_NAME ($(echo "$WHEEL_SIZE" | awk '{printf "%.1fMB", $1/1048576}'))"

if [ "$WHEEL_SIZE" -gt 10485760 ]; then
  echo "ERROR: Wheel exceeds 10MB Databricks Apps limit. Build it before yarn build."
  exit 1
fi

echo "==> Building JS frontend"
cd "$JS_DIR"
CI=false yarn build

echo "==> Packaging UI assets"
tar czf /tmp/build.tar.gz -C "$JS_DIR" build
UI_SIZE=$(stat -f%z /tmp/build.tar.gz 2>/dev/null || stat -c%s /tmp/build.tar.gz)
echo "    build.tar.gz ($(echo "$UI_SIZE" | awk '{printf "%.1fMB", $1/1048576}'))"

if [ "$UI_SIZE" -gt 10485760 ]; then
  echo "ERROR: UI assets exceed 10MB Databricks Apps limit."
  exit 1
fi

echo "==> Authenticating with Databricks (profile: $PROFILE)"
TOKEN=$(databricks auth token -p "$PROFILE" 2>&1 | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
HOST=$(databricks auth env -p "$PROFILE" --output json 2>&1 | python3 -c "import sys,json; print(json.load(sys.stdin)['env']['DATABRICKS_HOST'])")

# Ensure the app exists
if ! databricks apps get "$APP_NAME" -p "$PROFILE" > /dev/null 2>&1; then
  echo "ERROR: App '$APP_NAME' does not exist. Create it first with:"
  echo "  databricks apps create --json '{\"name\": \"$APP_NAME\"}' -p $PROFILE"
  exit 1
fi

WS_PATH=$(databricks apps get "$APP_NAME" -p "$PROFILE" -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['default_source_code_path'])")
echo "    Workspace path: $WS_PATH"

echo "==> Uploading wheel"
curl -sf -X POST "${HOST}/api/2.0/workspace/import" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "path=${WS_PATH}/${WHEEL_NAME}" \
  -F "format=AUTO" -F "overwrite=true" \
  -F "content=@${WHEEL_PATH}" \
  -o /dev/null
echo "    Uploaded $WHEEL_NAME"

echo "==> Uploading build.tar.gz"
curl -sf -X POST "${HOST}/api/2.0/workspace/import" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "path=${WS_PATH}/build.tar.gz" \
  -F "format=AUTO" -F "overwrite=true" \
  -F "content=@/tmp/build.tar.gz" \
  -o /dev/null
echo "    Uploaded build.tar.gz"

echo "==> Uploading app.py and app.yaml"
databricks workspace import "${WS_PATH}/app.py" \
  --file "$REPO_ROOT/.github/ui-preview/app.py" \
  --format AUTO --overwrite -p "$PROFILE"

# Generate app.yaml with resolved values
APP_URL=$(databricks apps get "$APP_NAME" -p "$PROFILE" -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['url'])")
PASSPHRASE=$(openssl rand -hex 32)
sed "s|__APP_URL__|${APP_URL}|; s|__KEK_PASSPHRASE__|${PASSPHRASE}|" \
  "$REPO_ROOT/.github/ui-preview/app.yaml" > /tmp/app.yaml
databricks workspace import "${WS_PATH}/app.yaml" \
  --file /tmp/app.yaml \
  --format AUTO --overwrite -p "$PROFILE"
rm -f /tmp/app.yaml

echo "==> Uploading requirements.txt"
echo "./${WHEEL_NAME}[genai]" > /tmp/requirements.txt
databricks workspace import "${WS_PATH}/requirements.txt" \
  --file /tmp/requirements.txt \
  --format AUTO --overwrite -p "$PROFILE"
rm -f /tmp/requirements.txt

echo "==> Deploying app"
databricks apps deploy "$APP_NAME" \
  --source-code-path "$WS_PATH" \
  -p "$PROFILE"

echo ""
echo "Deployed to: $APP_URL"
