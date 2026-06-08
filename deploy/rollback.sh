#!/usr/bin/env bash
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

set -euo pipefail
STATE_FILE="/home/jetson/recycleright/.deploy_state"
PREV_FILE="${STATE_FILE}.prev"

if [[ ! -f "$PREV_FILE" ]]; then
    echo "[rollback] No previous deployment found" >&2
    exit 1
fi

PREV=$(cat "$PREV_FILE")
echo "[rollback] Rolling back to tag=$PREV"
bash "$(dirname "$0")/deploy.sh" "$PREV"
