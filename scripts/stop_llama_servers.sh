#!/usr/bin/env bash
# Halt chat (:8000) and embedding (:8001) llama-server processes started by start_llama_servers.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log"
CHAT_PID_FILE="${LOG_DIR}/llama_chat.pid"
EMBED_PID_FILE="${LOG_DIR}/llama_embed.pid"
CHAT_PORT="${CHAT_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"

kill_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "No ${label} pid file (${pid_file})"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label} pid=${pid}"
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  else
    echo "${label} pid ${pid} not running"
  fi
  rm -f "${pid_file}"
}

kill_pid_file "${CHAT_PID_FILE}" "chat"
kill_pid_file "${EMBED_PID_FILE}" "embed"

# Best-effort cleanup if pid files were stale but ports still held
for port in "${CHAT_PORT}" "${EMBED_PORT}"; do
  if command -v fuser >/dev/null 2>&1; then
    if fuser "${port}/tcp" >/dev/null 2>&1; then
      echo "Freeing port ${port}"
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
  elif command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "Freeing port ${port} (pids: ${pids})"
      # shellcheck disable=SC2086
      kill ${pids} 2>/dev/null || true
    fi
  fi
done

echo "llama servers stopped."
