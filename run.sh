#!/bin/bash
set -o pipefail
umask 077

PROBLEM_FILE="$1"
TASK_ID="$2"
CHECKPOINT="$3"
REASONING_EFFORT="${4:-high}"
TEMPERATURE="$5"
TOP_P="$6"
MAX_OUTPUT_TOKENS="$7"

sampling=()
[ -n "$TEMPERATURE" ] && sampling+=(--temperature "$TEMPERATURE")
[ -n "$TOP_P" ] && sampling+=(--top-p "$TOP_P")
[ -n "$MAX_OUTPUT_TOKENS" ] && sampling+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")

/opt/post_training_agent/.venv/bin/python -m post_training_agent._permissions || exit 1
mkdir -p /logs/post_training_agent/atif
printf '%s' "{\"agent\":\"hackathon_post_training_agent\",\"model\":\"${CHECKPOINT}\",\"task_id\":\"${TASK_ID}\",\"log_path\":\"/logs/post_training_agent.log\"}" > /logs/agent_metadata.json

/opt/post_training_agent/.venv/bin/post-training-agent \
  --checkpoint "$CHECKPOINT" \
  --reasoning-effort "$REASONING_EFFORT" \
  "${sampling[@]}" \
  --problem-file "$PROBLEM_FILE" \
  --problem-dir /workspace \
  --task-id "$TASK_ID" \
  --output /logs/result.json 2>&1 | tee /logs/post_training_agent.log
status="${PIPESTATUS[0]}"

find /logs/post_training -name trajectory_atif.json -exec cp {} /logs/post_training_agent/atif/trajectory.json \;
exit "$status"
