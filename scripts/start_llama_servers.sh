#!/usr/bin/env bash
# Launch chat (Gemma :8000) and embedding (nomic-embed :8001) llama-server together.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log"
mkdir -p "${LOG_DIR}"

LLAMA_SERVER="${LLAMA_SERVER:-/home/sunny/AI/llama.cpp/build/bin/llama-server}"
CHAT_MODEL="${CHAT_MODEL:-/home/sunny/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf}"
EMBED_MODEL="${EMBED_MODEL:-/home/sunny/models/nomic-embed-text-v2-moe.Q8_0.gguf}"
CHAT_PORT="${CHAT_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"
NGL="${NGL:-99}"

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

echo "Starting chat server on :${CHAT_PORT} ..."
nohup "${LLAMA_SERVER}" \
  -m "${CHAT_MODEL}" \
  -ngl "${NGL}" \
  --host 0.0.0.0 \
  --port "${CHAT_PORT}" \
  >"${CHAT_LOG}" 2>&1 &
echo $! >"${CHAT_PID_FILE}"

echo "Starting embedding server on :${EMBED_PORT} ..."
# Larger physical batch avoids 500s on slightly longer news snippets (client also truncates).
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-2048}"
nohup "${LLAMA_SERVER}" \
  -m "${EMBED_MODEL}" \
  -ngl 0 \
  --host 0.0.0.0 \
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
  for i in $(seq 1 120); do
    if curl -sf --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "${name} ready: ${url}"
      return 0
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
echo "Both servers up:"
echo "  LLM_BASE_URL=http://127.0.0.1:${CHAT_PORT}/v1"
echo "  EMBEDDING_BASE_URL=http://127.0.0.1:${EMBED_PORT}/v1"
echo "  chat pid=$(cat "${CHAT_PID_FILE}")  log=${CHAT_LOG}"
echo "  embed pid=$(cat "${EMBED_PID_FILE}")  log=${EMBED_LOG}"
echo "Stop with: ${ROOT}/scripts/stop_llama_servers.sh"
