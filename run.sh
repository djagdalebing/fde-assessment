#!/usr/bin/env bash
# Launch any of the four services. Usage: ./run.sh <target>
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
PY="${PY:-$PWD/.venv/bin/python}"

usage() {
  cat <<'USAGE'
Usage: ./run.sh <target>

  task1                 MCP server on stdio (speaks JSON-RPC on stdout)
  task2-downstream      Mock downstream MCP server        :9002
  task2-gateway         MCP security gateway              :9001
  task3-provider        Mock LLM provider (SSE)           :9003
  task3-gateway         LLM gateway with PII guardrail    :9000
  task4-primary         Mock primary model provider       :9004
  task4-secondary       Mock secondary model provider     :9005
  task4-router          Rate-limiting / fallback router   :9006
  demo2 | demo3 | demo4 Start a task's servers and run a scripted demo
  test                  Run the whole test suite
USAGE
}

serve() { exec "$PY" -m uvicorn "$1" --host 127.0.0.1 --port "$2" --log-level "${LOG_LEVEL:-info}"; }

wait_for() {
  for _ in $(seq 1 60); do curl -sf "$1/healthz" >/dev/null 2>&1 && return 0; sleep 0.25; done
  echo "timed out waiting for $1" >&2; return 1
}

# A demo that binds nothing and then passes its health check against whatever
# was already listening is worse than one that fails: `set -euo pipefail` does
# not catch a failed background job, so a stale server on the port made the
# demo exercise an unrelated downstream and report success.
require_free_port() {
  if lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $1 is already in use - stop that process first, or the demo would" >&2
    echo "silently run against it instead of the server it means to start." >&2
    exit 1
  fi
}

case "${1:-}" in
  task1)            exec "$PY" -m task1_mcp_server ;;
  task2-downstream) serve task2_mcp_gateway.mock_downstream:app 9002 ;;
  task2-gateway)    export DOWNSTREAM_MCP_URL="${DOWNSTREAM_MCP_URL:-http://127.0.0.1:9002/mcp}"
                    serve task2_mcp_gateway.gateway:app 9001 ;;
  task3-provider)   serve task3_streaming_guardrail.mock_llm:app 9003 ;;
  task3-gateway)    export LLM_UPSTREAM_URL="${LLM_UPSTREAM_URL:-http://127.0.0.1:9003/v1/chat/completions}"
                    serve task3_streaming_guardrail.gateway:app 9000 ;;
  task4-primary)    MOCK_NAME=primary   serve task4_model_router.mock_entrypoints:primary 9004 ;;
  task4-secondary)  MOCK_NAME=secondary serve task4_model_router.mock_entrypoints:secondary 9005 ;;
  task4-router)     serve task4_model_router.app:app 9006 ;;
  test)             exec "$PY" -m pytest "${@:2}" ;;

  demo2)
    require_free_port 9002
    require_free_port 9001
    "$PY" -m uvicorn task2_mcp_gateway.mock_downstream:app --port 9002 --log-level warning & D=$!
    DOWNSTREAM_MCP_URL=http://127.0.0.1:9002/mcp \
      "$PY" -m uvicorn task2_mcp_gateway.gateway:app --port 9001 --log-level warning & G=$!
    trap 'kill $D $G 2>/dev/null || true' EXIT
    wait_for http://127.0.0.1:9002; wait_for http://127.0.0.1:9001
    echo "== viewer -> admin_reset_key (expect -32001, downstream never called)"
    curl -s -X POST http://127.0.0.1:9001/mcp -H "Authorization: Bearer viewer-token-def456" \
      -H 'Content-Type: application/json' \
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"acme"}}}'
    echo; echo "== admin -> admin_reset_key (expect result)"
    curl -s -X POST http://127.0.0.1:9001/mcp -H "Authorization: Bearer admin-token-abc123" \
      -H 'Content-Type: application/json' \
      -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"acme"}}}'
    echo; echo "== no credential (expect HTTP 401)"
    curl -s -o /dev/null -w 'HTTP %{http_code}\n' -X POST http://127.0.0.1:9001/mcp \
      -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":3,"method":"tools/list"}'
    ;;

  demo3)
    require_free_port 9003
    require_free_port 9000
    "$PY" -m uvicorn task3_streaming_guardrail.mock_llm:app --port 9003 --log-level warning & A=$!
    LLM_UPSTREAM_URL=http://127.0.0.1:9003/v1/chat/completions \
      "$PY" -m uvicorn task3_streaming_guardrail.gateway:app --port 9000 --log-level warning & B=$!
    trap 'kill $A $B 2>/dev/null || true' EXIT
    wait_for http://127.0.0.1:9003; wait_for http://127.0.0.1:9000
    echo "== streaming a response containing an email, credit card and SSN"
    curl -sN -X POST http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"m","stream":true,"scenario":"default","chunk_delay":0.05}' \
      | "$PY" -c '
import sys, json, time
start = time.time(); first = None; text = ""
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        break
    delta = json.loads(data)["choices"][0]["delta"].get("content")
    if delta:
        first = first if first is not None else time.time() - start
        text += delta
print("TTFT %.3fs   total %.3fs" % (first, time.time() - start))
print(text)'
    ;;

  demo4)
    require_free_port 9004
    require_free_port 9005
    require_free_port 9006
    rm -rf ./data && mkdir -p ./data
    MOCK_NAME=primary MOCK_DELAY=10 "$PY" -m uvicorn task4_model_router.mock_entrypoints:primary --port 9004 --log-level warning & P=$!
    MOCK_NAME=secondary "$PY" -m uvicorn task4_model_router.mock_entrypoints:secondary --port 9005 --log-level warning & S=$!
    ROUTER_DB_PATH=./data/router.db ROUTER_TOKENS_PER_MINUTE=300 \
      "$PY" -m uvicorn task4_model_router.app:app --port 9006 --log-level warning & R=$!
    trap 'kill $P $S $R 2>/dev/null || true' EXIT
    wait_for http://127.0.0.1:9004; wait_for http://127.0.0.1:9005; wait_for http://127.0.0.1:9006
    echo "== primary hangs for 10s; its deadline is 3000ms, so expect failover in ~3s"
    START=$(date +%s)
    curl -s -X POST http://127.0.0.1:9006/v1/chat/completions -H "Authorization: Bearer sk-tenant-acme-001" \
      -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":100}' \
      | "$PY" -c 'import sys,json; g=json.load(sys.stdin)["gateway"]; print("served_by:", g["served_by"], "attempts:", [a["outcome"] for a in g["attempts"]])'
    echo "elapsed: $(( $(date +%s) - START ))s"
    echo "== budget lowered to 300 tokens/min for the demo (the brief's figure is"
    echo "   50,000). Each request reserves prompt + max_tokens up front and settles"
    echo "   down to actual usage, so a few fit before the window is full."
    for i in $(seq 1 10); do
      code=$(curl -s -o ./data/resp.json -w '%{http_code}' -X POST http://127.0.0.1:9006/v1/chat/completions \
        -H "Authorization: Bearer sk-tenant-acme-001" -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":100}')
      echo "  request $i -> HTTP $code"
      [ "$code" = "429" ] && { "$PY" -m json.tool < ./data/resp.json; break; }
    done
    ;;

  *) usage; exit 1 ;;
esac
