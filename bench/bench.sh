#!/usr/bin/env bash
# Load-generation wrapper around `vllm bench serve`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults (override via env or CLI).
MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU_NAME="${GPU_NAME:-unknown-gpu}"
DTYPE="${DTYPE:-bf16}"
VLLM_VERSION="${VLLM_VERSION:-unknown}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REPEATS="${REPEATS:-3}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
REQUEST_RATES="${REQUEST_RATES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-vllm}"
DATASET_NAME="${DATASET_NAME:-random}"
SLO_TTFT_MS="${SLO_TTFT_MS:-}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
VLLM_FLAGS="${VLLM_FLAGS:-}"
METRICS_ENABLED="${METRICS_ENABLED:-1}"
METRICS_INTERVAL="${METRICS_INTERVAL:-0.5}"
METRICS_MAX_DURATION="${METRICS_MAX_DURATION:-3600}"
METRICS_FLUSH_SEC="${METRICS_FLUSH_SEC:-2}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bench.sh [options]

Environment variables (also accepted as --key value CLI overrides):
  MODEL, GPU_NAME, DTYPE, VLLM_VERSION, INPUT_LEN, OUTPUT_LEN, REPEATS,
  CONCURRENCIES, REQUEST_RATES, OUTPUT_DIR, NUM_PROMPTS, WARMUP_PROMPTS,
  HOST, PORT, BACKEND, DATASET_NAME, SLO_TTFT_MS, VLLM_API_KEY, VLLM_FLAGS,
  METRICS_ENABLED, METRICS_INTERVAL, METRICS_MAX_DURATION, METRICS_FLUSH_SEC

Options:
  --dry-run          Print commands without executing vllm bench serve.
  -h, --help         Show this help.
EOF
}

log() {
  printf '[bench] %s\n' "$*"
}

die() {
  printf '[bench] ERROR: %s\n' "$*" >&2
  exit 1
}

require_vllm() {
  if ! command -v vllm >/dev/null 2>&1; then
    die "'vllm' not found on PATH. Install vLLM or use --dry-run to verify command generation."
  fi
}

parse_flags_array() {
  if [[ -z "${VLLM_FLAGS// /}" ]]; then
    return 0
  fi
  # shellcheck disable=SC2206
  local flags=( $VLLM_FLAGS )
  printf '%s\n' "${flags[@]}"
}

write_meta() {
  local meta_path="$1"
  local mode="$2"
  local load_value="$3"
  local repeat="$4"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  local flags_json="[]"
  if [[ -n "$VLLM_FLAGS" ]]; then
    flags_json="["
    local first=1
    local flag
    while IFS= read -r flag; do
      [[ -z "$flag" ]] && continue
      if [[ "$first" -eq 1 ]]; then
        first=0
      else
        flags_json+=","
      fi
      flags_json+="\"${flag//\"/\\\"}\""
    done < <(parse_flags_array)
    flags_json+="]"
  fi

  local slo_line=""
  if [[ -n "$SLO_TTFT_MS" ]]; then
    slo_line=$(printf ', "slo_ttft_ms": %s' "$SLO_TTFT_MS")
  fi

  cat >"$meta_path" <<EOF
{
  "model": "$MODEL",
  "dtype": "$DTYPE",
  "gpu": "$GPU_NAME",
  "vllm_version": "$VLLM_VERSION",
  "flags": $flags_json,
  "input_len": $INPUT_LEN,
  "output_len": $OUTPUT_LEN,
  "mode": "$mode",
  "load_value": $load_value,
  "repeat": $repeat,
  "timestamp": "$timestamp"$slo_line
}
EOF
}

COMMON_ARGS=()

build_common_args() {
  COMMON_ARGS=(
    bench serve
    --backend "$BACKEND"
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL"
    --dataset-name "$DATASET_NAME"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --ignore-eos
    --metric-percentiles 50,95,99
    --percentile-metrics ttft,itl,e2el
    --disable-tqdm
  )
  if [[ -n "$SLO_TTFT_MS" ]]; then
    COMMON_ARGS+=(--goodput "ttft:${SLO_TTFT_MS}")
  fi
  if [[ -n "$VLLM_API_KEY" ]]; then
    COMMON_ARGS+=(--header "Authorization=Bearer ${VLLM_API_KEY}")
  fi
  local flag
  while IFS= read -r flag; do
    [[ -z "$flag" ]] && continue
    COMMON_ARGS+=("$flag")
  done < <(parse_flags_array)
}

metrics_url() {
  printf 'http://%s:%s/metrics' "$HOST" "$PORT"
}

start_metrics_sampler() {
  local metrics_file="$1"
  if [[ "$METRICS_ENABLED" != "1" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would start metrics sampler -> raw/$(basename "$metrics_file") duration=${METRICS_MAX_DURATION}s"
    return 0
  fi

  local -a sampler_cmd=(
    python3 "$SCRIPT_DIR/metrics_sampler.py"
    --duration "$METRICS_MAX_DURATION"
    --interval "$METRICS_INTERVAL"
    --url "$(metrics_url)"
    --out "$metrics_file"
  )
  if [[ -n "$VLLM_API_KEY" ]]; then
    sampler_cmd+=(--api-key "$VLLM_API_KEY")
  fi

  if ! "${sampler_cmd[@]}" >/dev/null & then
    log "WARNING: failed to start metrics sampler for $(basename "$metrics_file")"
    return 0
  fi
  echo $!
}

stop_metrics_sampler() {
  local sampler_pid="${1:-}"
  if [[ -z "$sampler_pid" || "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  if kill -0 "$sampler_pid" 2>/dev/null; then
    kill -TERM "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
  sleep "$METRICS_FLUSH_SEC"
}

run_vllm() {
  local description="$1"
  shift
  local -a cmd=(vllm "$@")
  log "$description"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  require_vllm
  "${cmd[@]}"
}

run_warmup() {
  local mode="$1"
  local load_value="$2"
  build_common_args
  local -a cmd=(vllm "${COMMON_ARGS[@]}" --num-prompts "$WARMUP_PROMPTS" --save-result)
  if [[ "$mode" == "max_concurrency" ]]; then
    cmd+=(--max-concurrency "$load_value")
  else
    cmd+=(--request-rate "$load_value")
  fi
  cmd+=(--result-dir "$RAW_DIR" --result-filename "_warmup_${mode}_${load_value}.json")
  run_vllm "warmup mode=$mode load=$load_value (discarded)" "${cmd[@]:1}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    rm -f "$RAW_DIR/_warmup_${mode}_${load_value}.json" \
          "$RAW_DIR/_warmup_${mode}_${load_value}.meta.json" 2>/dev/null || true
  fi
}

run_measured() {
  local mode="$1"
  local load_value="$2"
  local repeat="$3"
  local raw_file="${mode}_${load_value}_run${repeat}.json"
  local meta_file="${mode}_${load_value}_run${repeat}.meta.json"
  local metrics_file="$RAW_DIR/metrics_${mode}_${load_value}_run${repeat}.json"
  local sampler_pid=""

  sampler_pid="$(start_metrics_sampler "$metrics_file")"
  build_common_args
  local -a cmd=(
    vllm "${COMMON_ARGS[@]}"
    --num-prompts "$NUM_PROMPTS"
    --save-result
    --result-dir "$RAW_DIR"
    --result-filename "$raw_file"
  )
  if [[ "$mode" == "max_concurrency" ]]; then
    cmd+=(--max-concurrency "$load_value")
  else
    cmd+=(--request-rate "$load_value")
  fi
  if ! run_vllm "measured mode=$mode load=$load_value repeat=$repeat/$REPEATS -> raw/$raw_file" "${cmd[@]:1}"; then
    stop_metrics_sampler "$sampler_pid"
    return 1
  fi
  stop_metrics_sampler "$sampler_pid"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    write_meta "$RAW_DIR/$meta_file" "$mode" "$load_value" "$repeat"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpu-name|--gpu) GPU_NAME="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --input-len) INPUT_LEN="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
    --request-rates) REQUEST_RATES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --slo-ttft-ms) SLO_TTFT_MS="$2"; shift 2 ;;
    --vllm-flags) VLLM_FLAGS="$2"; shift 2 ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

RAW_DIR="$OUTPUT_DIR/raw"
mkdir -p "$RAW_DIR" "$OUTPUT_DIR/results"

log "config MODEL=$MODEL GPU=$GPU_NAME INPUT_LEN=$INPUT_LEN OUTPUT_LEN=$OUTPUT_LEN REPEATS=$REPEATS"
log "concurrencies=[$CONCURRENCIES] request_rates=[$REQUEST_RATES] dry_run=$DRY_RUN"

for c in $CONCURRENCIES; do
  run_warmup "max_concurrency" "$c"
  for repeat in $(seq 1 "$REPEATS"); do
    run_measured "max_concurrency" "$c" "$repeat"
  done
done

if [[ -n "$REQUEST_RATES" ]]; then
  for rate in $REQUEST_RATES; do
    run_warmup "request_rate" "$rate"
    for repeat in $(seq 1 "$REPEATS"); do
      run_measured "request_rate" "$rate" "$repeat"
    done
  done
fi

log "done"
