#!/usr/bin/env bash
# Wait until every GPU selected by a train/test shell script is free, then
# execute its screen payload in the same screen session used for waiting.

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash misc/wait_gpus_then_run_screen.sh TRAIN_SCRIPT [POLL_SECONDS]

TRAIN_SCRIPT must contain an active command of the form:
  screen -dmS NAME bash -c "CUDA_VISIBLE_DEVICES=0,1,... COMMAND"

and its last active line must be:
  screen -r NAME

POLL_SECONDS defaults to 60, or WAIT_GPU_POLL_SECONDS when that environment
variable is set. Only NVIDIA compute processes are considered; Xorg/desktop
graphics processes do not prevent launch.
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

trim_leading_space() {
    local value=$1
    printf '%s' "${value#"${value%%[![:space:]]*}"}"
}

last_active_line() {
    local file=$1 line trimmed last=''
    while IFS= read -r line || [[ -n $line ]]; do
        trimmed=$(trim_leading_space "$line")
        [[ -z $trimmed || $trimmed == \#* ]] && continue
        last=$line
    done < "$file"
    printf '%s\n' "$last"
}

parse_screen_name() {
    local file=$1 line
    line=$(last_active_line "$file")
    if [[ $line =~ ^[[:space:]]*screen[[:space:]]+-r[[:space:]]+([^[:space:]]+)[[:space:]]*$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    printf 'Error: last active line must be `screen -r NAME`, got: %s\n' "$line" >&2
    return 1
}

find_screen_launch_line() {
    local file=$1 screen_name=$2 line trimmed match=''
    while IFS= read -r line || [[ -n $line ]]; do
        trimmed=$(trim_leading_space "$line")
        [[ -z $trimmed || $trimmed == \#* ]] && continue
        if [[ $trimmed == *"screen -dmS ${screen_name} bash -c "* ]]; then
            match=$trimmed
        fi
    done < "$file"
    if [[ -z $match ]]; then
        printf 'Error: no active `screen -dmS %s bash -c ...` line found.\n' \
            "$screen_name" >&2
        return 1
    fi
    printf '%s\n' "$match"
}

extract_launch_payload() {
    local launch_line=$1 rest quote payload
    rest=${launch_line#*bash -c }
    if [[ $rest == "$launch_line" || ${#rest} -lt 2 ]]; then
        printf 'Error: cannot find `bash -c` payload in: %s\n' "$launch_line" >&2
        return 1
    fi

    quote=${rest:0:1}
    if [[ $quote != '"' && $quote != "'" ]]; then
        printf 'Error: bash -c payload must use matching single or double quotes.\n' >&2
        return 1
    fi
    if [[ ${rest: -1} != "$quote" ]]; then
        printf 'Error: bash -c payload must end with its opening quote.\n' >&2
        return 1
    fi
    payload=${rest:1:${#rest}-2}
    printf '%s\n' "$payload"
}

parse_gpu_csv() {
    local payload=$1 gpu_csv
    gpu_csv=$(printf '%s\n' "$payload" | sed -nE \
        "s/.*CUDA_VISIBLE_DEVICES=['\"]?([0-9]+(,[0-9]+)*)['\"]?.*/\1/p")
    if [[ -z $gpu_csv ]]; then
        printf 'Error: payload has no numeric CUDA_VISIBLE_DEVICES list: %s\n' \
            "$payload" >&2
        return 1
    fi
    printf '%s\n' "$gpu_csv"
}

apply_export_lines() {
    local file=$1 line trimmed assignment key value
    while IFS= read -r line || [[ -n $line ]]; do
        trimmed=$(trim_leading_space "$line")
        [[ -z $trimmed || $trimmed == \#* ]] && continue
        if [[ $trimmed =~ ^export[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key=${BASH_REMATCH[1]}
            value=${BASH_REMATCH[2]}
            # TRAIN_SCRIPT is user-controlled. Evaluate only the RHS so quoted
            # values behave exactly like a normal `export KEY=VALUE` line.
            eval "assignment=${value}"
            export "$key=$assignment"
            log "Inherited export from train script: $key"
        fi
    done < "$file"
}

gpu_compute_processes() {
    local gpu=$1
    nvidia-smi -i "$gpu" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

all_gpus_free() {
    local gpu_csv=$1 gpu output all_free=0
    IFS=',' read -r -a gpus <<< "$gpu_csv"
    for gpu in "${gpus[@]}"; do
        if ! output=$(gpu_compute_processes "$gpu"); then
            log "GPU $gpu query failed; treating it as unavailable."
            all_free=1
            continue
        fi
        if [[ -n ${output//[[:space:]]/} ]]; then
            log "GPU $gpu is occupied: ${output//$'\n'/; }"
            all_free=1
        else
            log "GPU $gpu is free."
        fi
    done
    return "$all_free"
}

worker_main() {
    local train_script=$1 poll_seconds=$2 screen_name=$3
    local launch_line launch_payload gpu_csv

    launch_line=$(find_screen_launch_line "$train_script" "$screen_name") || exit 2
    launch_payload=$(extract_launch_payload "$launch_line") || exit 2
    gpu_csv=$(parse_gpu_csv "$launch_payload") || exit 2
    apply_export_lines "$train_script"
    cd "$(dirname "$train_script")" || exit 2

    log "Waiting in screen '$screen_name' for GPUs: $gpu_csv"
    log "Polling every ${poll_seconds}s. Attach with: screen -r $screen_name"
    log "Pending command: $launch_payload"

    while ! all_gpus_free "$gpu_csv"; do
        log "At least one requested GPU is busy; checking again in ${poll_seconds}s."
        sleep "$poll_seconds"
    done

    log "All requested GPUs are free. Starting command in this screen."
    exec bash -c "$launch_payload"
}

main() {
    if [[ ${1:-} == '--worker' ]]; then
        [[ $# -eq 4 ]] || { usage >&2; exit 2; }
        worker_main "$2" "$3" "$4"
        return
    fi

    [[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
    command -v screen >/dev/null 2>&1 || {
        printf 'Error: `screen` is not installed or not in PATH.\n' >&2
        exit 2
    }
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf 'Error: `nvidia-smi` is not installed or not in PATH.\n' >&2
        exit 2
    }

    local train_script=$1
    local poll_seconds=${2:-${WAIT_GPU_POLL_SECONDS:-60}}
    [[ -f $train_script ]] || {
        printf 'Error: train script not found: %s\n' "$train_script" >&2
        exit 2
    }
    [[ $poll_seconds =~ ^[1-9][0-9]*$ ]] || {
        printf 'Error: POLL_SECONDS must be a positive integer.\n' >&2
        exit 2
    }

    train_script=$(realpath "$train_script")
    local screen_name self_path
    screen_name=$(parse_screen_name "$train_script") || exit 2
    self_path=$(realpath "${BASH_SOURCE[0]}")

    if screen -S "$screen_name" -Q select . >/dev/null 2>&1; then
        printf 'Error: screen session already exists: %s\n' "$screen_name" >&2
        printf 'Attach with: screen -r %s\n' "$screen_name" >&2
        exit 2
    fi

    if ! screen -dmS "$screen_name" bash "$self_path" \
        --worker "$train_script" "$poll_seconds" "$screen_name"; then
        printf 'Error: failed to create screen session: %s\n' "$screen_name" >&2
        exit 2
    fi
    printf 'Started GPU waiter in screen: %s\n' "$screen_name"
    printf 'Requested train script: %s\n' "$train_script"
    printf 'Attach with: screen -r %s\n' "$screen_name"
}

main "$@"
