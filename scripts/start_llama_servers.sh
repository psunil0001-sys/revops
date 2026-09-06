#!/usr/bin/env bash
# Launch chat (:8000) and embedding (:8001) llama-server together.
#
# Tuned defaults for AMD/Intel iGPU with ~16GB shared system RAM +
# Qwen*-9B Q4_K_M. Override any value via env.
#
# Why these numbers:
# - Qwen 9B Q4_K_M weights ≈ 5–6GB; iGPU shares RAM with OS/Streamlit/browser
# - -ngl 99 + -c 64000 OOMs Vulkan (radv DeviceLost) on typical 16GB boxes
# - RevOps forecast/briefs are fine with 4k context; partial GPU offload is safer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log"
mkdir -p "${LOG_DIR}"

LLAMA_SERVER="${LLAMA_SERVER:-/home/sunny/AI/llama.cpp/build/bin/llama-server}"
CHAT_MODEL="${CHAT_MODEL:-/home/sunny/models/Qwen3.5-9B-Q4_K_M.gguf}"
EMBED_MODEL="${EMBED_MODEL:-/home/sunny/models/nomic-embed-text-v2-moe.Q8_0.gguf}"
CHAT_PORT="${CHAT_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"

# iGPU / 16GB shared-RAM defaults (override as needed)
# NGL: start ~20; if stable and VRAM free, try 28–36. Use 0 for CPU-only.
NGL="${NGL:-20}"
# Context: 2048–4096 is enough for RevOps JSON forecast; 8192 max on 16GB iGPU
CHAT_CTX="${CHAT_CTX:-4096}"
# Parallel slots / batch — keep small on shared memory
CHAT_PARALLEL="${CHAT_PARALLEL:-1}"
CHAT_BATCH="${CHAT_BATCH:-512}"
CHAT_UBATCH="${CHAT_UBATCH:-128}"
# Threads for CPU-side work / leftover layers
CHAT_THREADS="${CHAT_THREADS:-$(nproc 2>/dev/null || echo 8)}"
# Bind localhost by default (safer than 0.0.0.0 without an API key)
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"

CHAT_PID_FILE="${LOG_DIR}/llama_chat.pid"
EMBED_PID_FILE="${LOG_DIR}/llama_embed.pid"
CHAT_LOG="${LOG_DIR}/llama_chat.log"
EMBED_LOG="${LOG_DIR}/llama_embed.log"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server not found or not executable: ${LLAMA_SERVER}"
  exit 1
fi

if [[ ! -f "${CHAT_MODEL}" ]]; then
  echo "ERROR: chat GGUF missing: ${CHAT_MODEL}"
  exit 1
fi

if [[ ! -f "${EMBED_MODEL}" ]]; then
  echo "ERROR: embedding GGUF missing: ${EMBED_MODEL}"
  echo "Place a nomic-embed GGUF at that path, or override EMBED_MODEL=/path/to/model.gguf"
  exit 1
fi

if [[ -f "${CHAT_PID_FILE}" ]] || [[ -f "${EMBED_PID_FILE}" ]]; then
  echo "PID files already exist under ${LOG_DIR}."
  echo "Stop first: ${ROOT}/scripts/stop_llama_servers.sh"
  exit 1
fi

echo "Starting chat server on ${LLAMA_HOST}:${CHAT_PORT} ..."
echo "  model=${CHAT_MODEL}"
echo "  ctx=${CHAT_CTX} ngl=${NGL} parallel=${CHAT_PARALLEL} batch=${CHAT_BATCH} threads=${CHAT_THREADS}"
nohup "${LLAMA_SERVER}" \
  -m "${CHAT_MODEL}" \
  -c "${CHAT_CTX}" \
  -ngl "${NGL}" \
  -np "${CHAT_PARALLEL}" \
  -b "${CHAT_BATCH}" \
  -ub "${CHAT_UBATCH}" \
  -t "${CHAT_THREADS}" \
  --host "${LLAMA_HOST}" \
  --port "${CHAT_PORT}" \
  >"${CHAT_LOG}" 2>&1 &
echo $! >"${CHAT_PID_FILE}"

echo "Starting embedding server on ${LLAMA_HOST}:${EMBED_PORT} ..."
# Keep embeddings on CPU so the iGPU budget goes to chat.
EMBED_NGL="${EMBED_NGL:-0}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
EMBED_CTX="${EMBED_CTX:-2048}"
nohup "${LLAMA_SERVER}" \
  -m "${EMBED_MODEL}" \
  -c "${EMBED_CTX}" \
  -ngl "${EMBED_NGL}" \
  --host "${LLAMA_HOST}" \
  --port "${EMBED_PORT}" \
  --embedding \
  --pooling mean \
  --batch-size "${EMBED_BATCH_SIZE}" \
  >"${EMBED_LOG}" 2>&1 &
echo $! >"${EMBED_PID_FILE}"

wait_ready() {
  local port="$1"
  local name="$2"
  local url="http://127.0.0.1:${port}/v1/models"
  local i
  for i in $(seq 1 180); do
    if curl -sf --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "${name} ready: ${url}"
      return 0
    fi
    # Surface early crashes (Vulkan DeviceLost, etc.)
    if [[ "${name}" == "chat" && -f "${CHAT_PID_FILE}" ]]; then
      local pid
      pid="$(cat "${CHAT_PID_FILE}")"
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: chat process exited early (pid ${pid}). Tail of ${CHAT_LOG}:"
        tail -n 40 "${CHAT_LOG}" || true
        return 1
      fi
    fi
    sleep 1
  done
  echo "ERROR: ${name} did not become ready on port ${port}."
  echo "See log under ${LOG_DIR}"
  return 1
}

wait_ready "${CHAT_PORT}" "chat" || exit 1
wait_ready "${EMBED_PORT}" "embeddings" || exit 1

echo ""
echo "Both servers up (16GB iGPU-safe defaults):"
echo "  LLM_BASE_URL=http://127.0.0.1:${CHAT_PORT}/v1"
echo "  EMBEDDING_BASE_URL=http://127.0.0.1:${EMBED_PORT}/v1"
echo "  chat pid=$(cat "${CHAT_PID_FILE}")  log=${CHAT_LOG}"
echo "  embed pid=$(cat "${EMBED_PID_FILE}")  log=${EMBED_LOG}"
echo "If chat still OOMs: export NGL=0  (CPU) or NGL=12 and retry."
echo "If stable with headroom: export NGL=28 CHAT_CTX=8192 and restart."
echo "Stop with: ${ROOT}/scripts/stop_llama_servers.sh"
