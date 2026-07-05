#!/bin/zsh
# 一键启动：Rust 后端 (server/) + React 前端 (frontend/)
# 双击运行或在终端执行；Ctrl+C 同时停止前后端。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_DIR="$ROOT_DIR/server"
FRONTEND_DIR="$ROOT_DIR/frontend"
SERVER_BIN="$SERVER_DIR/target/release/gomoku-server"
BACKEND_PORT=8000
FRONTEND_PORT=5173
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"
KATAGO_ROOT="${KATAGO_ROOT:-$ROOT_DIR/../KataGo}"
KATAGO_BIN="$KATAGO_ROOT/cpp/build-metal/katago"
KATAGO_MODEL="$KATAGO_ROOT/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz"
KATAGO_CONFIG="$KATAGO_ROOT/cpp/configs/gtp_example.cfg"

print_error() { echo "❌ $1" >&2; }
print_info()  { echo "▶ $1"; }

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    print_error "未找到命令: $1"
    exit 1
  fi
}

list_listening_pids() {
  lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null | awk 'NF && !seen[$0]++' || true
}

process_cwd() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1 || true
}

ensure_port_available_for_project() {
  local port="$1"
  local pids_raw pid cwd
  pids_raw="$(list_listening_pids "$port")"
  [[ -z "$pids_raw" ]] && return

  local pids=(${=pids_raw})
  for pid in "${pids[@]}"; do
    cwd="$(process_cwd "$pid")"
    if [[ -z "$cwd" || "$cwd" != "$ROOT_DIR"* ]]; then
      print_error "端口 $port 已被其他程序占用 (PID: $pid, CWD: ${cwd:-<unknown>})"
      exit 1
    fi
  done

  print_info "端口 $port 被本项目残留进程占用，正在自动清理: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  local attempt
  for attempt in {1..10}; do
    sleep 0.3
    [[ -z "$(list_listening_pids "$port")" ]] && return
  done
  pids_raw="$(list_listening_pids "$port")"
  if [[ -n "$pids_raw" ]]; then
    pids=(${=pids_raw})
    kill -9 "${pids[@]}" 2>/dev/null || true
    sleep 0.2
  fi
  if [[ -n "$(list_listening_pids "$port")" ]]; then
    print_error "端口 $port 清理失败，请手动处理后重试。"
    exit 1
  fi
}

require_command npm
require_command open

for f in "$KATAGO_BIN" "$KATAGO_MODEL" "$KATAGO_CONFIG"; do
  if [[ ! -e "$f" ]]; then
    print_error "缺少 KataGo 文件: $f"
    echo "请先在 $KATAGO_ROOT 完成 Metal 构建，或用 KATAGO_ROOT 环境变量指定位置。"
    exit 1
  fi
done

if [[ ! -x "$SERVER_BIN" ]]; then
  require_command cargo
  print_info "未找到后端二进制，正在编译 (cargo build --release)..."
  (cd "$SERVER_DIR" && cargo build --release)
fi

if [[ "${KATAGO_GUI_DRY_RUN:-0}" == "1" ]]; then
  print_info "DRY RUN: SERVER_BIN=$SERVER_BIN, FRONTEND_DIR=$FRONTEND_DIR"
  exit 0
fi

ensure_port_available_for_project "$BACKEND_PORT"
ensure_port_available_for_project "$FRONTEND_PORT"

print_info "项目目录: $ROOT_DIR"
print_info "后端 (Rust): $SERVER_BIN → 127.0.0.1:$BACKEND_PORT"
print_info "前端 (Vite): $FRONTEND_DIR → 127.0.0.1:$FRONTEND_PORT"

printf '\e]1;KataGo GUI\a'
cd "$ROOT_DIR"

echo "[KataGo GUI] 启动 Rust 后端..."
GOMOKU_REPO_ROOT="$ROOT_DIR" KATAGO_ROOT="$KATAGO_ROOT" "$SERVER_BIN" &
BACKEND_PID=$!

echo "[KataGo GUI] 启动前端..."
(
  cd "$FRONTEND_DIR" &&
  if [[ ! -d node_modules ]]; then npm install; fi &&
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "[KataGo GUI] 正在停止服务..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if [[ "${KATAGO_GUI_NO_BROWSER:-0}" != "1" ]]; then
  ( sleep 2; open "$FRONTEND_URL" ) >/dev/null 2>&1 &
fi

print_info "已启动。前端地址: $FRONTEND_URL"
wait $BACKEND_PID $FRONTEND_PID
