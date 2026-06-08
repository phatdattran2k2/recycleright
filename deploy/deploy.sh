#!/usr/bin/env bash
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

set -euo pipefail
TAG="${1:-latest}"
STATE_FILE="/home/jetson/recycleright/.deploy_state"
COMPOSE="docker compose -f /home/jetson/recycleright/deploy/docker-compose.yml"

# Save current tag for rollback
if [[ -f "$STATE_FILE" ]]; then
    PREV=$(cat "$STATE_FILE")
    echo "$PREV" > "${STATE_FILE}.prev"
fi
echo "$TAG" > "$STATE_FILE"

echo "[deploy] Pulling image tag=$TAG"
docker pull "ghcr.io/dinos611451001/recycleright:${TAG}"

echo "[deploy] Stopping current container"
$COMPOSE down || true

echo "[deploy] Starting new container"
TAG="$TAG" $COMPOSE up -d

echo "[deploy] Done — tag=$TAG"
