#!/usr/bin/env bash
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

set -euo pipefail
RETRIES=10
SLEEP=3

for i in $(seq 1 $RETRIES); do
    if docker ps --filter "name=recycleright" --filter "status=running" | grep -q recycleright; then
        echo "[healthcheck] Container is running (attempt $i)"
        exit 0
    fi
    echo "[healthcheck] Waiting... ($i/$RETRIES)"
    sleep $SLEEP
done

echo "[healthcheck] FAILED — container not running after ${RETRIES} attempts" >&2
exit 1