#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CPU_BUILD_DIR="${CPU_BUILD_DIR:-$SCRIPT_DIR/build_64_cpu}"
OPENCL_BUILD_DIR="${OPENCL_BUILD_DIR:-$SCRIPT_DIR/build_64_opencl}"
QNN_BUILD_DIR="${QNN_BUILD_DIR:-$SCRIPT_DIR/build_64_qnn}"
HOST_BUILD_DIR="${HOST_BUILD_DIR:-$ROOT_DIR/build_qnn_export}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/mnn-profiler}"
ADB_SERIAL="${ADB_SERIAL:-192.168.124.101:47954}"
MODEL_SRC_DIR="${MODEL_SRC_DIR:-/data/HUGGINGFACE/Qwen3-1.7B-MNN}"
QNN_EXPORT_SCRIPT="${QNN_EXPORT_SCRIPT:-$ROOT_DIR/transformers/llm/export/npu/generate_llm_qnn.py}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
QNN_SOC_ID="${QNN_SOC_ID:-69}"
QNN_DSP_ARCH="${QNN_DSP_ARCH:-v79}"
QNN_HEXAGON_ARCH="${QNN_HEXAGON_ARCH:-${QNN_DSP_ARCH#v}}"
BENCH_BIN="${BENCH_BIN:-}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/llm_profile_logs}"
WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/llm_profile_work}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

PROMPT_LENS=(128)        # 16 32 64 128 256 512 1024 2048 4096
DECODE_LENS=(1)          # 1 2 4 6 8 16 32 64
QNN_TEST_MODE="per-case" # per-case fixed-shape
QNN_FIXED_SHAPES=(1 8 128)
BACKENDS=(qnn) # cpu opencl qnn
WARMUP=2
REPEAT=3

CPU_LOG="$LOG_DIR/cpu.log"
OPENCL_LOG="$LOG_DIR/opencl.log"
QNN_LOG="$LOG_DIR/qnn.log"

ADB=(adb)
if [[ -n "$ADB_SERIAL" ]]; then
  ADB+=(-s "$ADB_SERIAL")
fi

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing file: $1" >&2
    exit 1
  fi
}

adb_run() {
  "${ADB[@]}" "$@"
}

copy_model_skeleton() {
  local src="$1"
  local dst="$2"
  rm -rf "$dst"
  mkdir -p "$dst"
  local files=(
    llm.mnn
    llm.mnn.weight
    llm_config.json
    tokenizer.txt
    config.json
  )
  local name
  for name in "${files[@]}"; do
    if [[ -f "$src/$name" ]]; then
      cp -f "$src/$name" "$dst/$name"
    fi
  done
}

patch_backend_type() {
  local config_path="$1"
  local backend_type="$2"
  python3 - "$config_path" "$backend_type" <<'PY'
import json
import sys

path, backend_type = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
data["backend_type"] = backend_type
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
    f.write("\n")
PY
}

sorted_unique_chunk_sizes() {
  local prompt_len="$1"
  local decode_len="$2"
  local values=("$prompt_len" "$decode_len" 1)
  local seen=""
  local filtered=()
  local value
  for value in "${values[@]}"; do
    value="${value//,/}"
    if [[ -z "$value" ]]; then
      continue
    fi
    if [[ ",$seen," != *",$value,"* ]]; then
      filtered+=("$value")
      seen+="${value},"
    fi
  done
  printf '%s\n' "${filtered[@]}" | sort -nr | tr '\n' ' '
}

join_by_comma() {
  local out=""
  local item
  for item in "$@"; do
    if [[ -n "$out" ]]; then
      out+=","
    fi
    out+="$item"
  done
  printf '%s' "$out"
}

join_by_underscore() {
  local out=""
  local item
  for item in "$@"; do
    if [[ -n "$out" ]]; then
      out+="_"
    fi
    out+="$item"
  done
  printf '%s' "$out"
}

set_array_from_list() {
  local -n target="$1"
  local raw="$2"
  local value
  raw="${raw//,/ }"
  target=()
  for value in $raw; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
      echo "invalid numeric list item: $value" >&2
      exit 1
    fi
    target+=("$value")
  done
  if [[ "${#target[@]}" -eq 0 ]]; then
    echo "empty numeric list is not allowed" >&2
    exit 1
  fi
}

set_string_array_from_list() {
  local -n target="$1"
  local raw="$2"
  local value
  raw="${raw//,/ }"
  target=()
  for value in $raw; do
    target+=("$value")
  done
  if [[ "${#target[@]}" -eq 0 ]]; then
    echo "empty list is not allowed" >&2
    exit 1
  fi
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --qnn-test-mode=per-case|fixed-shape
      per-case exports one QNN graph per prompt/decode pair.
      fixed-shape exports/reuses one graph with --qnn-fixed-shapes and runs all prompt/decode combinations.
  --qnn-fixed-shapes=1,128
      Chunk sizes used for the fixed-shape QNN export.
  --prompt-lens=512,1024,2048
  --decode-lens=1,2,4,6,8,16
  --backends=qnn,cpu,opencl
  --warmup=2
  --repeat=3
  -h, --help
USAGE
}

parse_args() {
  local arg value
  while [[ "$#" -gt 0 ]]; do
    arg="$1"
    case "$arg" in
    --qnn-test-mode=*)
      QNN_TEST_MODE="${arg#*=}"
      ;;
    --qnn-test-mode)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--qnn-test-mode requires a value" >&2
        exit 1
      }
      QNN_TEST_MODE="$1"
      ;;
    --qnn-fixed-shapes=*)
      set_array_from_list QNN_FIXED_SHAPES "${arg#*=}"
      ;;
    --qnn-fixed-shapes)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--qnn-fixed-shapes requires a value" >&2
        exit 1
      }
      set_array_from_list QNN_FIXED_SHAPES "$1"
      ;;
    --prompt-lens=*)
      set_array_from_list PROMPT_LENS "${arg#*=}"
      ;;
    --prompt-lens)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--prompt-lens requires a value" >&2
        exit 1
      }
      set_array_from_list PROMPT_LENS "$1"
      ;;
    --decode-lens=*)
      set_array_from_list DECODE_LENS "${arg#*=}"
      ;;
    --decode-lens)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--decode-lens requires a value" >&2
        exit 1
      }
      set_array_from_list DECODE_LENS "$1"
      ;;
    --backends=*)
      set_string_array_from_list BACKENDS "${arg#*=}"
      ;;
    --backends)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--backends requires a value" >&2
        exit 1
      }
      set_string_array_from_list BACKENDS "$1"
      ;;
    --warmup=*)
      value="${arg#*=}"
      [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "invalid --warmup: $value" >&2
        exit 1
      }
      WARMUP="$value"
      ;;
    --warmup)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--warmup requires a value" >&2
        exit 1
      }
      [[ "$1" =~ ^[0-9]+$ ]] || {
        echo "invalid --warmup: $1" >&2
        exit 1
      }
      WARMUP="$1"
      ;;
    --repeat=*)
      value="${arg#*=}"
      [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "invalid --repeat: $value" >&2
        exit 1
      }
      REPEAT="$value"
      ;;
    --repeat)
      shift
      [[ "$#" -gt 0 ]] || {
        echo "--repeat requires a value" >&2
        exit 1
      }
      [[ "$1" =~ ^[0-9]+$ ]] || {
        echo "invalid --repeat: $1" >&2
        exit 1
      }
      REPEAT="$1"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
    esac
    shift
  done

  case "$QNN_TEST_MODE" in
  per-case | fixed-shape)
    ;;
  *)
    echo "invalid --qnn-test-mode: $QNN_TEST_MODE" >&2
    exit 1
    ;;
  esac
}

resolve_bench_bin() {
  local build_dir="$1"
  local candidate
  if [[ -n "$BENCH_BIN" ]]; then
    if [[ -f "$build_dir/$BENCH_BIN" ]]; then
      printf '%s' "$BENCH_BIN"
      return
    fi
    echo "missing benchmark binary: $build_dir/$BENCH_BIN" >&2
    exit 1
  fi
  for candidate in profiler llm_profile_benchmark llm_profile; do
    if [[ -f "$build_dir/$candidate" ]]; then
      printf '%s' "$candidate"
      return
    fi
  done
  echo "cannot find profiler in $build_dir" >&2
  exit 1
}

backend_enabled() {
  local target="$1"
  local backend
  for backend in "${BACKENDS[@]}"; do
    if [[ "$backend" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

warn_qnn_python_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" != "mnn" ]]; then
    echo "QNN export uses Python tools. Please run: conda activate mnn" >&2
  fi
}

android_common_cmake_flags() {
  printf '%s\n' \
    -DMNN_BUILD_LLM=ON \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=true \
    -DMNN_OP_TIME_PROFILE=ON
}

configure_android_build() {
  local build_dir="$1"
  local backend="$2"
  local flags=()

  if [[ -z "${ANDROID_NDK:-}" ]]; then
    echo "ANDROID_NDK is required for Android build." >&2
    exit 1
  fi

  while IFS= read -r flag; do
    flags+=("$flag")
  done < <(android_common_cmake_flags)

  case "$backend" in
  cpu)
    flags+=(-DMNN_OPENCL=OFF -DMNN_QNN=OFF)
    ;;
  opencl)
    flags+=(-DMNN_OPENCL=ON -DMNN_QNN=OFF)
    ;;
  qnn)
    flags+=(-DMNN_OPENCL=OFF -DMNN_QNN=ON -DMNN_WITH_PLUGIN=ON)
    ;;
  esac

  mkdir -p "$build_dir"
  (cd "$build_dir" && "$SCRIPT_DIR/build_64.sh" "${flags[@]}")
}

ensure_android_built() {
  local build_dir="$1"
  local backend="$2"
  configure_android_build "$build_dir" "$backend"
  cmake --build "$build_dir" --target profiler -j"$BUILD_JOBS"
}

ensure_host_built() {
  local build_dir="$1"
  shift
  local target
  cmake -S "$ROOT_DIR" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS= \
    -DCMAKE_CXX_FLAGS= \
    -DMNN_BUILD_LLM=ON \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=true \
    -DMNN_OP_TIME_PROFILE=OFF \
    -DMNN_GPU_TIME_PROFILE=OFF \
    -DMNN_QNN=ON \
    -DMNN_QNN_CONVERT_MODE=ON \
    -DMNN_WITH_PLUGIN=ON \
    -DQNN_SDK_ROOT="$QNN_SDK_ROOT" >/dev/null
  for target in "$@"; do
    cmake --build "$build_dir" --target "$target" -j"$BUILD_JOBS"
  done
}

push_runtime_files() {
  local push_qnn="$1"
  local push_opencl="$2"
  local build_dir="$3"
  local files=(
    libMNN.so
    libMNN_Express.so
    libllm.so
    "$BENCH_BIN"
  )
  local file

  if [[ "$push_opencl" == "1" ]]; then
    files+=(libMNN_CL.so)
  fi

  adb_run shell "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'"

  for file in "${files[@]}"; do
    need_file "$build_dir/$file"
    adb_run push "$build_dir/$file" "$REMOTE_ROOT/" >/dev/null
  done

  if [[ "$push_qnn" == "1" ]]; then
    need_file "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so"
    need_file "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so"
    need_file "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so"
    need_file "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${QNN_HEXAGON_ARCH}Stub.so"
    need_file "${QNN_SDK_ROOT}/lib/hexagon-v${QNN_HEXAGON_ARCH}/unsigned/libQnnHtpV${QNN_HEXAGON_ARCH}Skel.so"

    adb_run push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "$REMOTE_ROOT/" >/dev/null
    adb_run push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "$REMOTE_ROOT/" >/dev/null
    adb_run push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "$REMOTE_ROOT/" >/dev/null
    adb_run push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${QNN_HEXAGON_ARCH}Stub.so" "$REMOTE_ROOT/" >/dev/null
    adb_run push "${QNN_SDK_ROOT}/lib/hexagon-v${QNN_HEXAGON_ARCH}/unsigned/libQnnHtpV${QNN_HEXAGON_ARCH}Skel.so" "$REMOTE_ROOT/" >/dev/null
  fi
}

run_remote_case() {
  local log_path="$1"
  local remote_dir="$2"
  local config_name="$3"
  local prompt_len="$4"
  local decode_len="$5"
  local enable_qnn="$6"
  local run_dir="$remote_dir"
  local config_path="$config_name"
  if [[ "$enable_qnn" == "1" ]]; then
    run_dir="$REMOTE_ROOT"
    config_path="$(basename "$remote_dir")/$config_name"
  fi
  local cmd="cd '$run_dir' && export LD_LIBRARY_PATH='$REMOTE_ROOT':\${LD_LIBRARY_PATH:-} && "
  if [[ "$enable_qnn" == "1" ]]; then
    cmd+="export ADSP_LIBRARY_PATH='$REMOTE_ROOT':\${ADSP_LIBRARY_PATH:-} && "
  fi
  adb_run shell "mkdir -p '$run_dir/tmp'"
  cmd+="'$REMOTE_ROOT/$BENCH_BIN' '$config_path' --prompt-lens=$prompt_len --decode-lens=$decode_len --warmup=$WARMUP --repeat=$REPEAT"
  adb_run shell "$cmd" 2>&1 | tee -a "$log_path"
}

run_cpu_or_opencl() {
  local backend="$1"
  local log_path="$2"
  local local_dir="$WORK_ROOT/${backend}_model"
  local remote_dir="$REMOTE_ROOT/$(basename "$local_dir")"
  local prompt_lens decode_lens

  copy_model_skeleton "$MODEL_SRC_DIR" "$local_dir"
  patch_backend_type "$local_dir/config.json" "$backend"

  adb_run push "$local_dir" "$REMOTE_ROOT/" >/dev/null

  printf '=== %s ===\n' "$backend" | tee -a "$log_path"
  prompt_lens="$(join_by_comma "${PROMPT_LENS[@]}")"
  decode_lens="$(join_by_comma "${DECODE_LENS[@]}")"
  printf '\n--- prompt=%s decode=%s ---\n' "$prompt_lens" "$decode_lens" | tee -a "$log_path"
  run_remote_case "$log_path" "$remote_dir" "config.json" "$prompt_lens" "$decode_lens" 0

  rm -rf "$local_dir"
  adb_run shell "rm -rf '$remote_dir'"
}

qnn_graph_ready() {
  local local_dir="$1"
  [[ -f "$local_dir/config_qnn.json" ]] || return 1
  [[ -d "$local_dir/qnn" ]] || return 1
  [[ -n "$(find "$local_dir/qnn" -type f -print -quit 2>/dev/null)" ]]
}

export_qnn_model() {
  local local_dir="$1"
  local cache_dir="$2"
  shift 2
  local chunk_sizes=("$@")

  export QNN_SDK_ROOT
  python3 "$QNN_EXPORT_SCRIPT" \
    --model "$local_dir" \
    --soc_id="$QNN_SOC_ID" \
    --dsp_arch="$QNN_DSP_ARCH" \
    --mnn_path="$HOST_BUILD_DIR" \
    --cache_path "$cache_dir" \
    --chunk_size "${chunk_sizes[@]}" 2>&1 | tee -a "$QNN_LOG"
}

run_qnn_case() {
  local prompt_len="$1"
  local decode_len="$2"
  local local_dir="$WORK_ROOT/qnn_p${prompt_len}_d${decode_len}"
  local remote_dir="$REMOTE_ROOT/$(basename "$local_dir")"
  local cache_dir="$local_dir/cache"
  local chunk_sizes=()
  local cleanup_local_qnn

  copy_model_skeleton "$MODEL_SRC_DIR" "$local_dir"
  cleanup_local_qnn() {
    rm -rf "$local_dir/qnn" "$local_dir/cache"
  }
  cleanup_local_qnn
  trap cleanup_local_qnn RETURN

  read -r -a chunk_sizes <<<"$(sorted_unique_chunk_sizes "$prompt_len" "$decode_len")"
  export_qnn_model "$local_dir" "$cache_dir" "${chunk_sizes[@]}"

  adb_run push "$local_dir" "$REMOTE_ROOT/" >/dev/null

  printf '\n--- prompt=%s decode=%s ---\n' "$prompt_len" "$decode_len" | tee -a "$QNN_LOG"
  run_remote_case "$QNN_LOG" "$remote_dir" "config_qnn.json" "$prompt_len" "$decode_len" 1

  cleanup_local_qnn
  rm -rf "$local_dir"
  adb_run shell "rm -rf '$remote_dir'"
  trap - RETURN
}

run_qnn_fixed_shape_case() {
  local shape_suffix
  shape_suffix="$(join_by_underscore "${QNN_FIXED_SHAPES[@]}")"
  local local_dir="$WORK_ROOT/qnn_fixed_s${shape_suffix}"
  local remote_dir="$REMOTE_ROOT/$(basename "$local_dir")"
  local cache_dir="$local_dir/cache"
  local prompt_lens decode_lens

  if qnn_graph_ready "$local_dir"; then
    printf '\n--- qnn fixed export shapes=%s: reuse existing graph ---\n' "${QNN_FIXED_SHAPES[*]}" | tee -a "$QNN_LOG"
  else
    printf '\n--- qnn fixed export shapes=%s: export graph ---\n' "${QNN_FIXED_SHAPES[*]}" | tee -a "$QNN_LOG"
    copy_model_skeleton "$MODEL_SRC_DIR" "$local_dir"
    rm -rf "$local_dir/qnn" "$local_dir/cache"
    export_qnn_model "$local_dir" "$cache_dir" "${QNN_FIXED_SHAPES[@]}"
  fi

  adb_run push "$local_dir" "$REMOTE_ROOT/" >/dev/null

  prompt_lens="$(join_by_comma "${PROMPT_LENS[@]}")"
  decode_lens="$(join_by_comma "${DECODE_LENS[@]}")"
  printf '\n--- fixed-shape prompt=%s decode=%s ---\n' "$prompt_lens" "$decode_lens" | tee -a "$QNN_LOG"
  run_remote_case "$QNN_LOG" "$remote_dir" "config_qnn.json" "$prompt_lens" "$decode_lens" 1

  adb_run shell "rm -rf '$remote_dir'"
}

main() {
  parse_args "$@"

  mkdir -p "$LOG_DIR" "$WORK_ROOT"
  need_file "$MODEL_SRC_DIR/llm.mnn"
  need_file "$MODEL_SRC_DIR/llm_config.json"
  need_file "$MODEL_SRC_DIR/config.json"

  if backend_enabled cpu; then
    ensure_android_built "$CPU_BUILD_DIR" cpu
    BENCH_BIN="$(resolve_bench_bin "$CPU_BUILD_DIR")"
    push_runtime_files 0 0 "$CPU_BUILD_DIR"
    run_cpu_or_opencl cpu "$CPU_LOG"
  fi

  if backend_enabled opencl; then
    ensure_android_built "$OPENCL_BUILD_DIR" opencl
    BENCH_BIN="$(resolve_bench_bin "$OPENCL_BUILD_DIR")"
    push_runtime_files 0 1 "$OPENCL_BUILD_DIR"
    run_cpu_or_opencl opencl "$OPENCL_LOG"
  fi

  if backend_enabled qnn; then
    if [[ -z "$QNN_SDK_ROOT" ]]; then
      echo "QNN_SDK_ROOT is required for qnn tests." >&2
      exit 1
    fi
    warn_qnn_python_env
    need_file "$QNN_EXPORT_SCRIPT"
    ensure_android_built "$QNN_BUILD_DIR" qnn
    ensure_host_built "$HOST_BUILD_DIR" generateLlmIO compilefornpu
    BENCH_BIN="$(resolve_bench_bin "$QNN_BUILD_DIR")"
    need_file "$HOST_BUILD_DIR/generateLlmIO"
    need_file "$HOST_BUILD_DIR/compilefornpu"
    push_runtime_files 1 0 "$QNN_BUILD_DIR"
    printf '=== qnn ===\n' | tee -a "$QNN_LOG"
    local prompt_len decode_len
    case "$QNN_TEST_MODE" in
    per-case)
      for prompt_len in "${PROMPT_LENS[@]}"; do
        for decode_len in "${DECODE_LENS[@]}"; do
          run_qnn_case "$prompt_len" "$decode_len"
        done
      done
      ;;
    fixed-shape)
      run_qnn_fixed_shape_case
      ;;
    esac
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
