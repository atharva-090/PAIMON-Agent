@echo off
REM Serve IFM/K2-Horizon-3.7B locally behind an OpenAI-compatible API on :8100.
REM Then point the agent at it with LLM_PROFILE=local (or pick
REM "brain: K2-Horizon (local)" in the web chat). The openai-mocker is
REM untouched -- switch back any time by dropping LLM_PROFILE.
REM
REM First run loads ~5B params in NF4 onto the GPU; expect a slow start.
cd /d "%~dp0"

REM --- Tunables. Sized for a 4 GB card; raise on a bigger GPU. -------------
if not defined K2_MODEL_PATH set K2_MODEL_PATH=E:\hf-models\K2-Horizon-3.7B
if not defined K2_QUANT set K2_QUANT=nf4
if not defined K2_MAX_CONTEXT set K2_MAX_CONTEXT=16384
if not defined K2_GPU_MEM set K2_GPU_MEM=3.2GiB
if not defined K2_REASONING_EFFORT set K2_REASONING_EFFORT=high
REM int4 KV is what makes high effort fit at 16k on a 4 GB card.
if not defined K2_KV_QUANT set K2_KV_QUANT=int4
if not defined K2_PORT set K2_PORT=8100

echo Starting K2-Horizon shim on http://127.0.0.1:%K2_PORT%/v1
echo   weights : %K2_MODEL_PATH%
echo   quant   : %K2_QUANT%   context: %K2_MAX_CONTEXT%   kv: %K2_KV_QUANT%
echo   reasoning: %K2_REASONING_EFFORT%
echo (Keep this window open. Press Ctrl+C to stop.)
".venv-k2\Scripts\python.exe" -m uvicorn local_model.serve_k2:app --host 127.0.0.1 --port %K2_PORT%
