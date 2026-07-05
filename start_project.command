#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"
KATAGO_ROOT="$ROOT_DIR/../KataGo"
KATAGO_BIN="$KATAGO_ROOT/cpp/build-metal/katago"
KATAGO_MODEL="$KATAGO_ROOT/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz"
KATAGO_CONFIG="$KATAGO_ROOT/cpp/configs/gtp_example.cfg"

print_error() {
  echo "❌ $1" >&2
}

print_info() {
  echo "▶ $1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    print_error "未找到命令: $1"
    exit 1
  fi
}

bootstrap_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi

  local candidates=(
    "$HOME/miniforge3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      if command -v conda >/dev/null 2>&1; then
        return
      fi
    fi
  done
}

ensure_port_free() {
  local port="$1"
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    print_error "端口 $port 已被占用，请先关闭已有服务后再启动。"
    exit 1
  fi
}

list_listening_pids() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NF && !seen[$0]++' || true
}

process_cwd() {
  local pid="$1"
  lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1 || true
}

process_cmd() {
  local pid="$1"
  ps -p "$pid" -o command= 2>/dev/null | sed 's/^ *//' || true
}

ensure_port_available_for_project() {
  local port="$1"
  local pids_raw pid cwd cmd
  pids_raw="$(list_listening_pids "$port")"
  if [[ -z "$pids_raw" ]]; then
    return
  fi

  local pids=(${=pids_raw})
  for pid in "${pids[@]}"; do
    cwd="$(process_cwd "$pid")"
    cmd="$(process_cmd "$pid")"
    if [[ -z "$cwd" || "$cwd" != "$ROOT_DIR"* ]]; then
      print_error "端口 $port 已被其他程序占用。"
      echo "PID: $pid"
      echo "CWD: ${cwd:-<unknown>}"
      echo "CMD: ${cmd:-<unknown>}"
      exit 1
    fi
  done

  print_info "端口 $port 被本项目残留进程占用，正在自动清理: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true

  local attempt
  for attempt in {1..10}; do
    sleep 0.3
    pids_raw="$(list_listening_pids "$port")"
    if [[ -z "$pids_raw" ]]; then
      return
    fi
  done

  pids=(${=pids_raw})
  print_info "端口 $port 上的残留进程未正常退出，执行强制清理: ${pids[*]}"
  kill -9 "${pids[@]}" 2>/dev/null || true
  sleep 0.2

  if [[ -n "$(list_listening_pids "$port")" ]]; then
    print_error "端口 $port 清理失败，请手动处理后重试。"
    exit 1
  fi
}

bootstrap_conda
require_command conda
require_command npm
require_command open

if [[ ! -f "$BACKEND_DIR/main.py" ]]; then
  print_error "未找到后端入口: $BACKEND_DIR/main.py"
  exit 1
fi

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  print_error "未找到前端入口: $FRONTEND_DIR/package.json"
  exit 1
fi

if [[ ! -x "$KATAGO_BIN" ]]; then
  print_error "未找到可执行的 KataGo: $KATAGO_BIN"
  echo "请先在 ../KataGo 完成 Metal 构建。"
  exit 1
fi

if [[ ! -f "$KATAGO_MODEL" ]]; then
  print_error "未找到 KataGo 模型: $KATAGO_MODEL"
  exit 1
fi

if [[ ! -f "$KATAGO_CONFIG" ]]; then
  print_error "未找到 KataGo 配置: $KATAGO_CONFIG"
  exit 1
fi

BACKEND_DIR_Q="$(printf '%q' "$BACKEND_DIR")"
FRONTEND_DIR_Q="$(printf '%q' "$FRONTEND_DIR")"
ROOT_DIR_Q="$(printf '%q' "$ROOT_DIR")"

COMBINED_CMD="$(cat <<EOF
printf '\e]1;KataGo GUI\a'
cd ${ROOT_DIR_Q} || exit 1

echo "[KataGo GUI] 启动后端..."
(
  cd ${BACKEND_DIR_Q} &&
  conda run -n base uvicorn main:app --reload --host 127.0.0.1 --port ${BACKEND_PORT}
) &
BACKEND_PID=\$!

echo "[KataGo GUI] 启动前端..."
(
  cd ${FRONTEND_DIR_Q} &&
  if [ ! -d node_modules ]; then
    npm install
  fi &&
  npm run dev -- --host 127.0.0.1 --port ${FRONTEND_PORT}
) &
FRONTEND_PID=\$!

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "[KataGo GUI] 正在停止服务..."
  kill \$BACKEND_PID \$FRONTEND_PID 2>/dev/null || true
  wait \$BACKEND_PID \$FRONTEND_PID 2>/dev/null || true
}

trap cleanup INT TERM EXIT
wait \$BACKEND_PID \$FRONTEND_PID
EOF
)"

if [[ "${KATAGO_GUI_DRY_RUN:-${GOMOKU_DRY_RUN:-0}}" == "1" ]]; then
  print_info "ROOT_DIR=${ROOT_DIR_Q}"
  print_info "COMBINED_CMD=${COMBINED_CMD}"
  exit 0
fi

if ! conda run -n base python -c "import fastapi, uvicorn, websockets, pydantic" >/dev/null 2>&1; then
  print_error "conda base 环境缺少 Web 依赖。请先运行："
  echo "conda run -n base python -m pip install -r \"$ROOT_DIR/requirements.txt\""
  exit 1
fi

ensure_port_available_for_project "$BACKEND_PORT"
ensure_port_available_for_project "$FRONTEND_PORT"

print_info "项目目录: $ROOT_DIR"
print_info "KataGo: $KATAGO_BIN"
print_info "启动后端: $BACKEND_PORT"
print_info "启动前端: $FRONTEND_PORT"
print_info "终端模式: 当前窗口运行，Ctrl+C 可同时停止前后端"

if [[ "${KATAGO_GUI_NO_BROWSER:-${GOMOKU_NO_BROWSER:-0}}" != "1" ]]; then
  (
    sleep 2
    open "${FRONTEND_URL}"
  ) >/dev/null 2>&1 &
fi

print_info "已发起启动。前端地址: ${FRONTEND_URL}"
eval "$COMBINED_CMD"
