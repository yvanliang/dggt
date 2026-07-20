#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# 从 26 主节点直接执行：
#   bash kill_four_nodes.sh
#
# 这是强制清理脚本：会在 25/26/30/31 上立即 SIGKILL 所有匹配的
# train_scene_flow_pretrain.py / torch.distributed.run / torchrun 进程，
# 包括可能同时运行的两节点或其它同名训练任务。

MODE="${1:---all}"

WORKER_USER="wuzn"
SSH_PORT=2288

NODE_HOSTS=(
    "10.199.7.26"
    "10.199.7.25"
    "10.199.7.30"
    "10.199.7.31"
)
WORKER_RANKS=(1 2 3)

LIANGYY_ROOT="${LIANGYY_ROOT:-/home/wuzn/liangyy}"
PROJECT_ROOT="${PROJECT_ROOT:-${LIANGYY_ROOT}/dggt}"
LAUNCH_LOG_DIR="${PROJECT_ROOT}/logs/distributed_launch_four_nodes"

SSH_OPTS=(
    -p "${SSH_PORT}"
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=no
)

node_host_for_rank() {
    local rank="$1"

    if [[ ! "${rank}" =~ ^[0-9]+$ ]] || (( rank < 0 || rank >= ${#NODE_HOSTS[@]} )); then
        echo "[错误] node_rank 无效：${rank}" >&2
        return 1
    fi

    echo "${NODE_HOSTS[rank]}"
}

kill_local_training() {
    local rank="$1"
    local host
    local pid_file="${LAUNCH_LOG_DIR}/node_rank${rank}.pid"

    host="$(node_host_for_rank "${rank}")"
    echo "[rank ${rank} ${host}] 强制终止所有匹配的训练进程……"

    # 方括号写法可避免 pkill 匹配到它自身的命令行。
    pkill -KILL -f '[t]rain_scene_flow_pretrain.py' 2>/dev/null || true
    pkill -KILL -f '[t]orch.distributed.run' 2>/dev/null || true
    pkill -KILL -f '[t]orchrun' 2>/dev/null || true

    rm -f -- "${pid_file}"
    echo "[rank ${rank} ${host}] 清理完成。"
}

kill_remote_training() {
    local rank="$1"
    local host="$2"
    local pid_file="${LAUNCH_LOG_DIR}/node_rank${rank}.pid"

    ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${host}" \
        "pkill -KILL -f '[t]rain_scene_flow_pretrain.py' 2>/dev/null || true;
         pkill -KILL -f '[t]orch.distributed.run' 2>/dev/null || true;
         pkill -KILL -f '[t]orchrun' 2>/dev/null || true;
         rm -f -- '${pid_file}';
         echo '[rank ${rank} ${host}] 清理完成。'"
}

case "${MODE}" in
    --all)
        overall_rc=0

        # 先杀远端 worker，最后杀本机 master。
        for rank in "${WORKER_RANKS[@]}"; do
            host="$(node_host_for_rank "${rank}")"
            echo "[rank ${rank}] 连接 ${host}……"
            if ! kill_remote_training "${rank}" "${host}"; then
                echo "[错误] rank ${rank} ${host} 清理失败。" >&2
                overall_rc=1
            fi
        done

        if ! kill_local_training 0; then
            overall_rc=1
        fi

        if [[ "${overall_rc}" -eq 0 ]]; then
            echo "[OK] 25/26/30/31 上匹配的训练进程已全部强制终止。"
        else
            echo "[错误] 部分节点清理失败，请查看上方错误信息。" >&2
        fi
        exit "${overall_rc}"
        ;;

    --local)
        TARGET_RANK="${2:?用法：bash ${0} --local <node_rank>}"
        kill_local_training "${TARGET_RANK}"
        ;;

    *)
        echo "未知参数：${MODE}" >&2
        echo "用法：bash ${0}" >&2
        exit 2
        ;;
esac
