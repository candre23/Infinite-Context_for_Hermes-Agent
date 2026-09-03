#!/usr/bin/env bash
set -euo pipefail

REPO="${HERMES_REPO:-${HOME}/.hermes/hermes-agent}"
PLUGIN_ROOT="${REPO}/plugins/context_engine"
ENGINE_DIR="${PLUGIN_ROOT}/infinite_v0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [[ ! -d "${REPO}" ]]; then
  echo "Hermes Agent repository not found: ${REPO}" >&2
  echo "Set HERMES_REPO if Hermes is installed elsewhere." >&2
  exit 1
fi

PY=""
for candidate in \
  "${REPO}/venv/bin/python" \
  "${REPO}/.venv/bin/python" \
  "${HOME}/.hermes/.venv/bin/python" \
  "$(command -v python3 || true)"; do
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    PY="${candidate}"
    break
  fi
done

if [[ -z "${PY}" ]]; then
  echo "Could not locate the Hermes/Python interpreter." >&2
  exit 1
fi

echo "Using Python: ${PY}"
"${PY}" -m py_compile "${SCRIPT_DIR}/infinite_v0/__init__.py"

if [[ "${HERMES_INFINITE_SKIP_EMBEDDINGS:-0}" != "1" ]]; then
  if ! "${PY}" -c 'import fastembed' >/dev/null 2>&1; then
    echo "Installing FastEmbed for local semantic retrieval..."
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "${PY}" 'fastembed>=0.7,<0.8'
    else
      "${PY}" -m pip install 'fastembed>=0.7,<0.8'
    fi
  else
    echo "FastEmbed already available in Hermes runtime."
  fi

  echo "Checking BAAI/bge-small-en-v1.5..."
  "${PY}" -c 'from fastembed import TextEmbedding; m=TextEmbedding(model_name="BAAI/bge-small-en-v1.5"); next(iter(m.embed(["Infinite Context semantic backend readiness test"]))); print("Semantic backend ready.")'
else
  echo "Skipping FastEmbed because HERMES_INFINITE_SKIP_EMBEDDINGS=1."
fi

mkdir -p "${PLUGIN_ROOT}"
if [[ -d "${ENGINE_DIR}" ]]; then
  BACKUP="${ENGINE_DIR}.backup-${STAMP}"
  cp -a "${ENGINE_DIR}" "${BACKUP}"
  echo "Backed up existing plugin to: ${BACKUP}"
fi

mkdir -p "${ENGINE_DIR}"
cp "${SCRIPT_DIR}/infinite_v0/__init__.py" "${ENGINE_DIR}/__init__.py"
cp "${SCRIPT_DIR}/infinite_v0/plugin.yaml" "${ENGINE_DIR}/plugin.yaml"

# The trace is diagnostic-only; start a fresh file after an upgrade.
rm -f "${HOME}/.hermes/context_engine/infinite_v0_trace.log"

echo
echo "Installed Infinite Context v0.9.2 to: ${ENGINE_DIR}"
echo "Existing SQLite data, memories, project bindings, and embeddings were preserved."
echo "Ensure Hermes config selects context.engine: infinite_v0, then restart Hermes."
echo "After restart, run /infinite status."
