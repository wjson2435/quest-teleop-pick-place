#!/usr/bin/env bash
# Diffusion Policy training (LeRobot) on the teleop dataset.
#
# Usage:
#   ./train.sh dp_pick_place_v1                        # full run -> pick_place/outputs/dp_pick_place_v1
#   STEPS=500 SAVE_FREQ=500 ./train.sh dp_smoke2       # short test run (checks the pipeline)
#   DATASET=panda_pick_place_joint_position ./train.sh dp_pos_v1   # another dataset in pick_place/datasets
#   ./train.sh dp_v2 --policy.horizon=32               # extra options are passed to lerobot-train
#                                                      # (don't repeat the ones set below)
#
# The run name must be new: lerobot-train stops if the output folder already exists.
#
# Model (lerobot defaults): input = last 2 frames (0.1 s) of the wrist/front images + observation.state,
# output = 16 actions (0.8 s, joint targets + gripper), of which 8 (0.4 s) are executed.

set -euo pipefail

RUN_NAME="${1:?usage: ./train.sh <run_name> [extra lerobot-train options]}"
shift

DATASET="${DATASET:-panda_pick_place_joint_impedance}"
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

LEROBOT_TRAIN="$HOME/anaconda3/envs/mujo/bin/lerobot-train"  # conda env "mujo"
HERE="$(dirname "$(realpath "$0")")"  # pick_place/
DATASET_ROOT="$HERE/datasets/$DATASET"
OUTPUT_DIR="$HERE/outputs/$RUN_NAME"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "$OUTPUT_DIR already exists, choose another run name."
  exit 1
fi

"$LEROBOT_TRAIN" \
  --dataset.repo_id="local/$DATASET" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.crop_shape="[216,288]" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --num_workers="$NUM_WORKERS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable=False \
  "$@"
