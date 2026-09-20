#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
    echo "Usage: bash tools/test.sh TASK CHECKPOINT [test.py options]" >&2
    echo "Tasks: s3dis_det scannet_det arkitscenes_det multiscan_det 3rscan_det scannetpp_det scannet_seg s3dis_seg" >&2
    exit 2
fi
TASK="$1"
CHECKPOINT="$2"
shift 2
case "$TASK" in
    s3dis_det|scannet_det|arkitscenes_det|multiscan_det|3rscan_det|scannetpp_det|scannet_seg|s3dis_seg) ;;
    *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
# Preserve relative checkpoint paths supplied from another working directory.
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
cd "$ROOT_DIR"
exec "${PYTHON:-python}" tools/test.py "configs/${TASK}.py" "$CHECKPOINT" "$@"
