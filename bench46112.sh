#!/usr/bin/env bash
# E2E A/B benchmark for vllm-project/vllm PR #46112
# (decode-only fast path in Scheduler.update_from_output).
#
# ONE EDITABLE INSTALL, BOTH SIDES -- verified, not assumed:
#   git diff 55c98e370 scheduler-update-from-output-opt --stat
#     benchmarks/overheads/benchmark_scheduler_update_from_output.py
#     tests/v1/core/test_scheduler.py
#     vllm/v1/core/sched/scheduler.py
#   Three .py files, no C++/CUDA/cmake. *.so is gitignored (.gitignore:33) so the
#   VLLM_USE_PRECOMPILED artifacts survive every checkout, and vllm/_version.py is
#   gitignored (.gitignore:2) so setuptools-scm does not rewrite the torch.compile
#   cache key between sides. So `git checkout` alone really does swap the code.
#   BUT that is only true if `vllm` on PATH actually imports from $REPO, and the
#   original script never checked. This one proves it twice:
#     (1) once at startup: the interpreter behind the `vllm` console script must
#         import vllm from $REPO;
#     (2) before every single server start: md5 of the on-disk scheduler.py must
#         equal `git show <ref>:vllm/v1/core/sched/scheduler.py`.
#   Silently benchmarking the wrong code is the worst outcome here, so both are fatal.
#
# INSTALL EXACTLY ONCE, ON THE NAMED BRANCH. setup.py's precompiled-wheel picker
# runs `git branch --show-current` (setup.py:909); in detached HEAD that is empty,
# merge-base throws, and it silently falls back to the "nightly" wheel whose .so
# files can be far ahead of 55c98e370. Never reinstall between sides either: that
# rewrites vllm/_version.py, which is a compile-cache factor, so one side would
# pay a cold compile the other did not.
#
# Ordering is ABBA (base,pr / pr,base / ...) so within-rep thermal ramp on a rented
# 5090 is not systematically attributed to one side.
#
#   bash bench46112.sh
#   RUNNERS=1 DECODE_REPS=1 PREFILL_REPS=1 bash bench46112.sh   # end-to-end smoke
#   EXTRA_SERVE_ARGS="--no-async-scheduling" bash bench46112.sh # unoverlapped arm
#
# No `set -e` on purpose: pipefail + a non-matching grep would otherwise abort a
# two-hour run. Instead, everything that can silently corrupt the comparison is
# checked explicitly and aborts loudly.

set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
OUT="${OUT:-$HOME/bench46112}"
REPO="${REPO:-$HOME/vllm}"
BASE_REF="${BASE_REF:-55c98e370}"
PR_REF="${PR_REF:-scheduler-update-from-output-opt}"
DECODE_REPS="${DECODE_REPS:-8}"
PREFILL_REPS="${PREFILL_REPS:-2}"
# VLLM_USE_V2_MODEL_RUNNER is parsed by maybe_convert_bool = bool(int(v))
# (vllm/envs.py:324-327), so the ONLY meaningful values are 1 (V2) and 0 (V1).
# "2" would also be True -- that is the bug that made the old script measure V2
# twice and file half of it under "V1". V1 is the arm the has_num_nans_in_logits
# gate fix is actually about, so it must really be 0.
RUNNERS="${RUNNERS:-1 0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_UTIL="${GPU_UTIL:-0.85}"
DECODE_CONC="${DECODE_CONC:-256}"
# 1024 prompts x 1024 out tokens = 4 full waves at concurrency 256, ~60-90s of
# measurement per server start. 500 was ~2 waves / ~30s, too short for a 5% effect.
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
PREFILL_PROMPTS="${PREFILL_PROMPTS:-512}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1200}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

mkdir -p "$OUT"
cd "$REPO" || { echo "FATAL: REPO not found: $REPO"; exit 1; }
REPO_ABS="$(pwd -P)"

log()   { echo "[$(date +%H:%M:%S)] $*"; }
fatal() { log "FATAL: $*"; exit 1; }

# Set by run_side; read by bench() for --metadata provenance in the result JSON.
SIDE_NAME="init"; SIDE_SHA="none"; SIDE_V2="1"
# Real value is measured just before the warm-up; initialised here so that an
# early ^C running stop_server from the trap does not trip `set -u`.
GPU_MEM_BASELINE=0

# ---------------------------------------------------------------- preflight --

for t in git curl md5sum nvidia-smi awk; do
  command -v "$t" >/dev/null || fatal "missing required tool: $t"
done

START_REF="$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)"
finish() { git checkout -q "$START_REF" 2>/dev/null; rm -f "$REPO_ABS/benchmarks/overheads/_micro_ab.py"; }
trap 'echo; log "interrupted"; stop_server; finish; exit 130' INT TERM

# A tracked local edit (e.g. to scheduler.py, exactly the file a dev iterating on
# this PR would have poked) makes `git checkout` refuse -- which was previously
# non-fatal and silently dropped one whole side. Untracked files are harmless.
[ -z "$(git status --porcelain --untracked-files=no)" ] \
  || fatal "worktree has tracked modifications; commit or stash first (git checkout would fail mid-run)"

git rev-parse --verify -q "$BASE_REF^{commit}" >/dev/null || fatal "BASE_REF not reachable: $BASE_REF"
git rev-parse --verify -q "$PR_REF^{commit}"   >/dev/null || fatal "PR_REF not reachable: $PR_REF"

FREE_GB="$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1048576)}')"
[ "${FREE_GB:-0}" -ge 20 ] \
  || fatal "only ${FREE_GB}G free on \$HOME; need ~20G for the HF snapshot + ~/.cache/vllm"

# Install once, on the NAMED branch (see header).
git checkout -q "$PR_REF" || fatal "cannot checkout $PR_REF"
if [ "$SKIP_INSTALL" != "1" ]; then
  log "editable install (once, on $PR_REF) -- do NOT rerun this between sides"
  if command -v uv >/dev/null; then
    VLLM_USE_PRECOMPILED=1 uv pip install -e . || fatal "editable install failed"
  else
    VLLM_USE_PRECOMPILED=1 pip install -e . || fatal "editable install failed"
  fi
fi

# The `vllm` console script (pyproject.toml:43) resolves from the venv's bin dir,
# NOT from cwd, so `cd $REPO` does not shadow a preinstalled wheel. Ask the
# interpreter that actually runs `vllm serve` where it imports vllm from.
VLLM_BIN="$(command -v vllm)" || fatal "no 'vllm' on PATH"
PYBIN="$(sed -n '1s/^#!//p' "$VLLM_BIN" | awk '{print $1}')"
case "$PYBIN" in */python*) : ;; *) PYBIN="$(command -v python3)" ;; esac
log "vllm console script: $VLLM_BIN  (interpreter: $PYBIN)"

"$PYBIN" - "$REPO_ABS" <<'PYEOF' || fatal "vllm does not import from \$REPO -- git checkout would be a no-op"
import os, sys
repo = os.path.realpath(sys.argv[1])
import vllm
p = os.path.realpath(vllm.__file__)
want = os.path.join(repo, "vllm") + os.sep
if not p.startswith(want):
    sys.stderr.write(f"vllm imports from {p}, expected under {want}\n")
    sys.exit(1)
print(f"OK: vllm -> {p}  (version {vllm.__version__})")
PYEOF

# Per-ref fingerprint of the one file the PR actually changes at runtime. This is
# the only artifact that can prove after the fact which code produced a result:
# vllm/_version.py is baked at install time, so vllm.__version__ (and the version
# recorded in every result JSON) is byte-identical for base and pr by construction.
BASE_SCHED_MD5="$(git show "$BASE_REF:vllm/v1/core/sched/scheduler.py" | md5sum | awk '{print $1}')"
PR_SCHED_MD5="$(git show "$PR_REF:vllm/v1/core/sched/scheduler.py"   | md5sum | awk '{print $1}')"
[ "$BASE_SCHED_MD5" != "$PR_SCHED_MD5" ] \
  || fatal "scheduler.py is identical on $BASE_REF and $PR_REF -- there is nothing to A/B"
log "scheduler.py md5: base=$BASE_SCHED_MD5 pr=$PR_SCHED_MD5"

gpu_mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

# ---------------------------------------------------------------- functions --

stop_server() {
  # EngineCore and the workers rename themselves via setproctitle to
  # "VLLM::EngineCore" / "VLLM::Worker" (vllm/v1/engine/core.py:1289 ->
  # vllm/utils/system_utils.py:198), so their cmdline no longer contains
  # "vllm serve". Matching only "vllm serve" orphans the process that holds all
  # the VRAM, and the pkill -9 fallback orphans it permanently.
  pkill -f 'vllm serve' 2>/dev/null
  pkill -f 'VLLM::'     2>/dev/null
  local i used
  for i in $(seq 1 90); do
    pgrep -f 'vllm serve|VLLM::' >/dev/null || break
    sleep 1
  done
  pkill -9 -f 'VLLM::'     2>/dev/null
  pkill -9 -f 'vllm serve' 2>/dev/null
  # Do not return until the GPU has actually given the memory back, or the next
  # --gpu-memory-utilization 0.85 profile run either OOMs or silently sizes a
  # smaller KV cache -- which would quietly change that rep's achievable concurrency.
  for i in $(seq 1 60); do
    used="$(gpu_mem_used)"
    if [ "${used:-999999}" -le "$((GPU_MEM_BASELINE + 1024))" ]; then return 0; fi
    sleep 2
  done
  log "GPU still holding ${used} MiB (baseline ${GPU_MEM_BASELINE} MiB) after stop_server"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
  return 1
}

assert_runner() {  # $1=v2flag  $2=tag
  local v2="$1" tag="$2" saw=0
  # gpu_worker.py:384-385 logs "Using V2 Model Runner" ONLY in the V2 branch;
  # there is no V1 counterpart anywhere in the tree, so absence is the V1 signal.
  # Two-sided, and fatal -- the old one-sided grep could never fail.
  grep -q "Using V2 Model Runner" "$OUT/server_${tag}.log" && saw=1
  echo "$tag wanted_v2=$v2 observed_v2=$saw" >> "$OUT/runner.txt"
  if [ "$v2" != "$saw" ]; then
    log "RUNNER MISMATCH ($tag): asked for VLLM_USE_V2_MODEL_RUNNER=$v2, log says v2=$saw"
    return 1
  fi
  log "runner confirmed ($tag): v2=$saw"
  return 0
}

start_server() {  # $1=v2flag  $2=tag
  local v2="$1" tag="$2" pid deadline
  # A leftover server answering on $PORT would make the new (failed-to-bind)
  # process invisible and every rep would benchmark one unchanging build.
  if curl -sf --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "port $PORT is already serving before launch ($tag) -- refusing to start"
    return 1
  fi
  log "starting server ($tag, VLLM_USE_V2_MODEL_RUNNER=$v2)"
  VLLM_USE_V2_MODEL_RUNNER="$v2" nohup vllm serve "$MODEL" \
    --port "$PORT" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --no-enable-prefix-caching \
    --disable-uvicorn-access-log \
    $EXTRA_SERVE_ARGS \
    > "$OUT/server_${tag}.log" 2>&1 &
  pid=$!
  deadline=$((SECONDS + SERVER_TIMEOUT_S))
  while [ "$SECONDS" -lt "$deadline" ]; do
    # Bound to the PID we launched, not to whatever happens to hold the port.
    if ! kill -0 "$pid" 2>/dev/null; then
      log "SERVER DIED ($tag) -- tail of $OUT/server_${tag}.log:"
      tail -40 "$OUT/server_${tag}.log"
      return 1
    fi
    # --max-time: /health awaits EngineCore over ZMQ and can block forever if the
    # engine wedges; without it the "timeout" is not a timeout.
    if curl -sf --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      log "server up ($tag)"
      assert_runner "$v2" "$tag" || return 1
      return 0
    fi
    sleep 2
  done
  log "SERVER TIMEOUT ($tag) after ${SERVER_TIMEOUT_S}s -- tail of log:"
  tail -40 "$OUT/server_${tag}.log"
  return 1
}

bench() {  # $1=tag $2=inlen $3=outlen $4=concurrency $5=num_prompts
  local tag="$1" inlen="$2" outlen="$3" conc="$4" np="$5"
  local t0=$SECONDS rc ok nfail
  vllm bench serve \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --dataset-name random \
    --random-input-len "$inlen" \
    --random-output-len "$outlen" \
    --num-prompts "$np" \
    --max-concurrency "$conc" \
    --ignore-eos \
    --seed 0 \
    --disable-tqdm \
    --ready-check-timeout-sec 120 \
    --num-warmups 32 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metadata side="$SIDE_NAME" sha="$SIDE_SHA" runner_v2="$SIDE_V2" \
    --save-result --result-filename "$OUT/${tag}.json" \
    > "$OUT/${tag}.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "BENCH FAILED ($tag) rc=$rc"
    tail -25 "$OUT/${tag}.txt"
    return $rc
  fi
  # A run with failed requests still prints a plausible throughput number.
  ok="$(awk '/^Successful requests:/{print $NF}'  "$OUT/${tag}.txt")"
  nfail="$(awk '/^Failed requests:/{print $NF}'   "$OUT/${tag}.txt")"
  if [ "${ok:-x}" != "$np" ] || [ "${nfail:-x}" != "0" ]; then
    log "BAD RUN ($tag): successful=$ok failed=$nfail expected=$np"
    return 1
  fi
  log "$tag ok in $((SECONDS - t0))s"
  return 0
}

run_side() {  # $1=ref $2=side $3=v2flag $4=rep
  local ref="$1" side="$2" v2="$3" rep="$4" want got rname tag
  git checkout -q "$ref" || { log "git checkout $ref failed"; return 1; }
  if [ "$side" = base ]; then want="$BASE_SCHED_MD5"; else want="$PR_SCHED_MD5"; fi
  got="$(md5sum "$REPO_ABS/vllm/v1/core/sched/scheduler.py" | awk '{print $1}')"
  [ "$want" = "$got" ] || { log "scheduler.py on disk ($got) != $ref ($want)"; return 1; }

  SIDE_NAME="$side"
  SIDE_SHA="$(git rev-parse --short HEAD)"
  SIDE_V2="$v2"
  rname=V1; [ "$v2" = 1 ] && rname=V2
  tag="${rname}_${side}_rep${rep}"
  log "--- $side @ $SIDE_SHA  runner=$rname  scheduler.py=$got  tag=$tag"
  echo "$tag side=$side sha=$(git rev-parse HEAD) scheduler_md5=$got" >> "$OUT/provenance.txt"

  start_server "$v2" "$tag" || { stop_server; return 1; }
  # Decode-bound: the arm the PR is expected to improve (~+5% at conc 256).
  bench "decode_${tag}" 128 1024 "$DECODE_CONC" "$NUM_PROMPTS" || { stop_server; return 1; }
  # Prefill-bound: regression guard for when the GPU dominates.
  if [ "$rep" -le "$PREFILL_REPS" ]; then
    bench "prefill_${tag}_c128" 512 64 128 "$PREFILL_PROMPTS" || { stop_server; return 1; }
    bench "prefill_${tag}_c256" 512 64 256 "$PREFILL_PROMPTS" || { stop_server; return 1; }
  fi
  stop_server || return 1
}

summarize() {  # $1 = "COMPLETE" | "<reason the run is incomplete>"
  local f t name
  : > "$OUT/summary.csv"
  echo "phase,runner,side,rep,out_tok_s,mean_tpot_ms,mean_ttft_ms,successful,server_peak_gen_tok_s" >> "$OUT/summary.csv"
  echo
  echo "=== decode + prefill  (output tok/s, mean TPOT ms, mean TTFT ms, server-side gen tok/s) ==="
  for f in "$OUT"/decode_*.txt "$OUT"/prefill_*.txt; do
    [ -e "$f" ] || continue
    name="$(basename "$f" .txt)"
    t="${name#decode_}"; t="${t#prefill_}"; t="${t%_c128}"; t="${t%_c256}"
    # NOTE: "Peak output token throughput" is lowercase-o (serve.py:1199) so it
    # does NOT collide with "Output token throughput" (serve.py:1194); the anchor
    # just keeps that true if either string is ever renamed.
    printf '%-36s %10s %10s %10s %10s\n' "$name" \
      "$(grep -E '^Output token throughput' "$f" | awk '{print $NF}')" \
      "$(grep -E '^Mean TPOT' "$f" | awk '{print $NF}')" \
      "$(grep -E '^Mean TTFT' "$f" | awk '{print $NF}')" \
      "$(grep -o 'Avg generation throughput: [0-9.]*' "$OUT/server_${t}.log" 2>/dev/null | awk '{print $NF}' | sort -g | tail -1)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(echo "$name" | cut -d_ -f1)" \
      "$(echo "$t" | cut -d_ -f1)" "$(echo "$t" | cut -d_ -f2)" "$(echo "$t" | cut -d_ -f3)" \
      "$(grep -E '^Output token throughput' "$f" | awk '{print $NF}')" \
      "$(grep -E '^Mean TPOT' "$f" | awk '{print $NF}')" \
      "$(grep -E '^Mean TTFT' "$f" | awk '{print $NF}')" \
      "$(grep -E '^Successful requests' "$f" | awk '{print $NF}')" \
      "$(grep -o 'Avg generation throughput: [0-9.]*' "$OUT/server_${t}.log" 2>/dev/null | awk '{print $NF}' | sort -g | tail -1)" \
      >> "$OUT/summary.csv"
  done
  echo
  echo "CPU-only control (update_from_output microbenchmark, no GPU):"
  tail -6 "$OUT/micro_base.txt" 2>/dev/null | sed 's/^/  base | /'
  tail -6 "$OUT/micro_pr.txt"   2>/dev/null | sed 's/^/  pr   | /'
  echo
  echo "machine-readable: $OUT/summary.csv    provenance: $OUT/provenance.txt    runners: $OUT/runner.txt"
  if [ "$1" != COMPLETE ]; then
    echo
    echo "***********************************************************************"
    echo "*** $1"
    echo "*** RUN IS INCOMPLETE -- do NOT compare means across sides as-is.   ***"
    echo "***********************************************************************"
  fi
}

# ---------------------------------------------------------------- warm-up ----

GPU_MEM_BASELINE="$(gpu_mem_used)"; GPU_MEM_BASELINE="${GPU_MEM_BASELINE:-0}"
log "GPU memory baseline: ${GPU_MEM_BASELINE} MiB"
stop_server >/dev/null 2>&1
curl -sf --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  && fatal "something is still serving on port $PORT; kill it before starting"

log "prefetching $MODEL"
"$PYBIN" -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" \
  >/dev/null 2>&1 || fatal "could not prefetch $MODEL"

# CPU-only positive control. async_scheduling defaults to True for this config
# (vllm/config/vllm.py:1143), so update_from_output overlaps the forward pass and
# a flat E2E result does NOT by itself refute the PR. This microbenchmark measures
# the CPU cost directly, costs no GPU time, and distinguishes "fast path never
# engaged" from "engaged but hidden by overlap".
MICRO_SRC="benchmarks/overheads/benchmark_scheduler_update_from_output.py"
MICRO="$REPO_ABS/benchmarks/overheads/_micro_ab.py"   # untracked; absent from both refs
if git cat-file -e "$PR_REF:$MICRO_SRC" 2>/dev/null; then
  git show "$PR_REF:$MICRO_SRC" > "$MICRO"
  for side in base pr; do
    if [ "$side" = base ]; then r="$BASE_REF"; else r="$PR_REF"; fi
    git checkout -q "$r" || fatal "cannot checkout $r"
    log "CPU control: update_from_output microbenchmark on $side ($(git rev-parse --short HEAD))"
    "$PYBIN" "$MICRO" --num-requests 32 128 256 --iters 200 > "$OUT/micro_${side}.txt" 2>&1 \
      || log "WARNING: microbenchmark failed on $side (non-fatal) -- see $OUT/micro_${side}.txt"
    tail -6 "$OUT/micro_${side}.txt"
  done
  rm -f "$MICRO"
  git checkout -q "$PR_REF" || fatal "cannot checkout $PR_REF"
fi

# Prime the HF cache, the inductor/compile cache and CUDA-graph capture for EACH
# runner before the timed matrix. VLLM_USE_V2_MODEL_RUNNER is a compile-cache
# factor (it is NOT in envs.py's ignored_factors), so V1 and V2 get separate cache
# dirs and both need warming. This also fails early if V1 cannot come up at all.
for v2 in $RUNNERS; do
  rname=V1; [ "$v2" = 1 ] && rname=V2
  SIDE_NAME=warmup; SIDE_SHA="$(git rev-parse --short HEAD)"; SIDE_V2="$v2"
  start_server "$v2" "warmup_${rname}" || { stop_server; finish; fatal "warmup server failed for runner $rname"; }
  bench "warmup_${rname}" 128 256 64 128 || { stop_server; finish; fatal "warmup bench failed for runner $rname"; }
  stop_server || { finish; fatal "GPU did not release memory after warmup ($rname)"; }
done

# ---------------------------------------------------------------- main loop --

log "repo=$REPO base=$BASE_REF pr=$PR_REF model=$MODEL out=$OUT"
log "matrix: $(echo $RUNNERS | wc -w) runner(s) x $DECODE_REPS reps x 2 sides = $(( $(echo $RUNNERS | wc -w) * DECODE_REPS * 2 )) server starts, expect ~2h"
log "^C is safe: it stops the server and restores $START_REF"

EXPECTED_DECODE=$(( $(echo $RUNNERS | wc -w) * DECODE_REPS * 2 ))

for v2 in $RUNNERS; do
  rname=V1; [ "$v2" = 1 ] && rname=V2
  for rep in $(seq 1 "$DECODE_REPS"); do
    log "=== runner=$rname rep $rep/$DECODE_REPS ==="
    # ABBA, not ABAB: fixed base-then-pr order would load every within-rep drift
    # (fan/clock ramp, noisy neighbour) onto the pr side.
    if [ $((rep % 2)) -eq 1 ]; then order="base pr"; else order="pr base"; fi
    for side in $order; do
      if [ "$side" = base ]; then r="$BASE_REF"; else r="$PR_REF"; fi
      if ! run_side "$r" "$side" "$v2" "$rep"; then
        stop_server
        finish
        summarize "run_side failed: runner=$rname side=$side rep=$rep"
        exit 1
      fi
    done
  done
done

finish
stop_server >/dev/null 2>&1

GOT_DECODE=$(ls "$OUT"/decode_*.txt 2>/dev/null | wc -l)
log "done. raw results in $OUT"
if [ "$GOT_DECODE" -eq "$EXPECTED_DECODE" ]; then
  summarize COMPLETE
else
  summarize "only $GOT_DECODE/$EXPECTED_DECODE decode results present"
  exit 1
fi
