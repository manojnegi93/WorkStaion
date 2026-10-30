# Run the whole app on http://localhost:8000 with no model downloaded (offline demo mode).
# Needs Python 3.12. Uses the built UI in frontend/dist, so Node is not required.
$backend = Join-Path $PSScriptRoot "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    python -m venv (Join-Path $backend ".venv")
    & $python -m pip install -r (Join-Path $backend "requirements.txt")
}
$env:LLM_PROVIDER = "offline"
$env:EMBEDDING_PROVIDER = "hashing"
$env:QDRANT_PATH = ":memory:"
Set-Location $backend
& $python -m uvicorn app.main:app --port 8000
